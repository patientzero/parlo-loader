"""Conservative CHAT reader. Raw source always survives interpretation."""
import re
import unicodedata

BULLET = re.compile(r'\x15([^\x15]*)\x15')
TIME = re.compile(r'^(-?\d+)_(-?\d+)$')
PLACEHOLDER = re.compile(r'\b(?:xxx|yyy|www)\b', re.I)


def clean_text(text):
    """Remove structural CHAT codes, retaining spoken fillers and repetitions."""
    text = BULLET.sub('', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\((?:\d+(?:\.\d+)?|\.+)\)', '', text)
    text = re.sub(r'&[=][^\s]+|\+[\w/.!?]+|\b0\w*', '', text)
    text = re.sub(r'&[-~]', '', text)
    text = re.sub(r'@\w+(?::\w+)?', '', text)
    text = text.translate(str.maketrans('', '', '<>'))
    return ' '.join(text.split())


def normalize_text(text, *, lowercase=True, punctuation=True):
    text = unicodedata.normalize('NFC', text)
    if lowercase:
        text = text.lower()
    if punctuation:
        text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.split())


def bounds(start, end, duration_ms):
    if start is None or end is None:
        return 'missing'
    if start < 0 or end <= start or end > duration_ms:
        return 'invalid'
    return 'valid'


def parse_chat(source, duration_ms):
    issues, tiers = [], []
    for n, line in enumerate(source.split('\n'), 1):
        line = line.rstrip('\r')
        if not line.strip():
            continue
        if line.startswith(('@', '*', '%')):
            label, sep, body = line.partition(':')
            tiers.append({'label': label, 'text': body.strip() if sep else '', 'line': n})
        elif line[:1].isspace() and tiers:
            tiers[-1]['text'] += '\n' + line.strip()
        else:
            issues.append(('malformed_line', f'Line {n}: unrecognized CHAT line'))
    headers = [t for t in tiers if t['label'].startswith('@')]
    roles = {}
    for h in headers:
        if h['label'] == '@ID':
            fields = h['text'].split('|')
            if len(fields) > 7:
                roles[fields[2].strip()] = fields[7].strip().lower()
    notrans = any(h['label'] == '@Media' and 'notrans' in h['text'].lower() for h in headers)
    if not any(h['label'] == '@End' for h in headers):
        issues.append(('missing_end', 'Missing @End header'))
    utterances, words = [], []
    current = None
    for t in tiers:
        if t['label'].startswith('*'):
            speaker = t['label'][1:]
            role = roles.get(speaker, '')
            participant = role in {'subject', 'participant', 'target_child', 'patient'} or (not role and speaker in {'PAR', 'CHI'})
            markers = BULLET.findall(t['text'])
            parsed = [TIME.fullmatch(m) for m in markers]
            start = end = None
            if len(parsed) == 1 and parsed[0]:
                start, end = map(int, parsed[0].groups())
            timing = bounds(start, end, duration_ms)
            if markers and (len(parsed) != 1 or not all(parsed)) or t['text'].count('\x15') % 2:
                timing = 'invalid'
            if timing == 'invalid':
                issues.append(('invalid_utterance_timing', f'Line {t["line"]}: invalid/ambiguous timing'))
            text = clean_text(t['text'])
            placeholder = notrans or bool(PLACEHOLDER.search(text))
            current = dict(utterance_index=len(utterances), speaker=speaker, is_participant=participant,
                           raw_text=t['text'], text=text, start_ms=start, end_ms=end,
                           timing_status=timing, is_placeholder=placeholder,
                           valid_reference=participant and not placeholder and any(c.isalpha() for c in text),
                           dependent_tiers=[], line=t['line'])
            utterances.append(current)
        elif t['label'].startswith('%'):
            if current is None:
                issues.append(('orphan_tier', f'Line {t["line"]}: dependent tier without speaker'))
                continue
            current['dependent_tiers'].append(t)
            if t['label'] != '%wor':
                continue
            # Each bullet belongs only to the immediately preceding lexical token.
            # Earlier tokens without bullets retain null timings.
            if t['text'].count('\x15') % 2:
                issues.append(('malformed_word_timing', f'Line {t["line"]}: unclosed timing marker'))
            tokens = re.findall(r'\x15[^\x15]*\x15|\[[^\]]*\]|\((?:\d+(?:\.\d+)?|\.+)\)|[^\s\x15]+', t['text'])
            pending = None
            for token in tokens:
                if token.startswith('\x15'):
                    match = TIME.fullmatch(token.strip('\x15'))
                    if pending is None:
                        issues.append(('orphan_word_timing', f'Line {t["line"]}: timing without word'))
                        continue
                    if match:
                        pending['start_ms'], pending['end_ms'] = map(int, match.groups())
                    pending['timing_status'] = bounds(pending['start_ms'], pending['end_ms'], duration_ms) if match else 'invalid'
                    if pending['timing_status'] == 'valid' and current['timing_status'] == 'valid':
                        if pending['start_ms'] < current['start_ms'] or pending['end_ms'] > current['end_ms']:
                            pending['timing_status'] = 'invalid'
                    if pending['timing_status'] == 'invalid':
                        issues.append(('invalid_word_timing', f'Line {t["line"]}: invalid word timing'))
                    pending = None
                else:
                    cleaned = clean_text(token)
                    if not any(c.isalnum() for c in cleaned):
                        continue
                    pending = dict(utterance_index=current['utterance_index'], word_index=len(words),
                                   speaker=current['speaker'], is_participant=current['is_participant'],
                                   raw_text=token, text=cleaned, start_ms=None, end_ms=None, timing_status='missing')
                    words.append(pending)
    if not utterances:
        issues.append(('missing_utterances', 'No speaker turns'))
    # Preserve words even when a transcript has no %wor tier.
    covered = {w['utterance_index'] for w in words}
    for u in utterances:
        if u['utterance_index'] not in covered:
            for token in u['text'].split():
                if any(c.isalnum() for c in token):
                    words.append(dict(utterance_index=u['utterance_index'], word_index=len(words), speaker=u['speaker'],
                                      is_participant=u['is_participant'], raw_text=token, text=token,
                                      start_ms=None, end_ms=None, timing_status='missing'))
    return headers, utterances, words, issues
