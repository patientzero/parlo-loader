from parlo.chat import parse_chat, clean_text, normalize_text


def test_multiline_speakers_and_raw():
    source = '@Begin\n@ID:\tdeu|P|S|||||Subject||\n*INV:\tHallo\n*S:\t&-äh ich [/] ich\n\tgehe . \x1510_800\x15\n%wor:\täh \x1510_100\x15 ich ich \x15200_300\x15 gehe \x15300_800\x15 .\n%com:\tfirst\n\tsecond\n@End\n'
    headers, turns, words, issues = parse_chat(source, 1000)
    assert not issues
    assert [t['is_participant'] for t in turns] == [False, True]
    assert turns[1]['text'] == 'äh ich ich gehe .'
    assert '\n' in turns[1]['raw_text']
    assert turns[1]['dependent_tiers'][1]['text'] == 'first\nsecond'
    assert [w['start_ms'] for w in words if w['is_participant']] == [10, None, 200, 300]
    assert headers[1]['label'] == '@ID'


def test_placeholders_and_notrans():
    for text in ['xxx .', 'ein xxx Wort', 'yyy .', 'www .']:
        _, turns, _, _ = parse_chat(f'@Begin\n*PAR:\t{text}\n@End\n', 1000)
        assert turns[0]['is_placeholder'] and not turns[0]['valid_reference']
    _, turns, _, _ = parse_chat('@Media:\tx, audio, notrans\n*CHI:\tWort\n@End\n', 1000)
    assert not turns[0]['valid_reference']


def test_invalid_and_missing_boundaries():
    for marker in ['-1_100', '200_100', '0_1001', '0_0', 'broken']:
        _, turns, words, issues = parse_chat(f'*PAR:\tWort \x15{marker}\x15\n%wor:\tWort \x15{marker}\x15\n@End\n', 1000)
        assert turns[0]['timing_status'] == 'invalid'
        assert words[0]['timing_status'] == 'invalid'
        assert issues
    _, turns, words, _ = parse_chat('*PAR:\tWort\n@End\n', 1000)
    assert turns[0]['start_ms'] is None and words[0]['start_ms'] is None


def test_malformed_lines_and_orphan_tiers():
    _, _, _, issues = parse_chat('%com:\torphan\nbroken\n*PAR:\tHallo', 1000)
    assert {'malformed_line', 'orphan_tier', 'missing_end'} <= {i[0] for i in issues}


def test_normalization_preserves_fillers_repetitions():
    assert clean_text('ähm ich [/] ich <will> (.) &-äh &~hm .') == 'ähm ich ich will äh hm .'
    assert normalize_text('Ähm, ich ich!') == 'ähm ich ich'
    assert normalize_text('Ähm!', lowercase=False, punctuation=False) == 'Ähm!'


def test_word_annotations_and_malformed_bullets():
    _, _, words, issues = parse_chat('*PAR:\tich komme \x1510_200\x15\n%wor:\tich [=! whispered softly] \x1510_100\x15 (.) komme \x15100_200\x15\n@End\n',1000)
    assert [w['text'] for w in words] == ['ich','komme']
    assert [w['start_ms'] for w in words] == [10,100]
    assert not issues
    _, _, _, issues = parse_chat('*PAR:\tWort\n%wor:\tWort \x1510_20\n@End\n',1000)
    assert 'malformed_word_timing' in {code for code,_ in issues}
