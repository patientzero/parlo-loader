import csv
import numpy as np
import pytest
import soundfile as sf

CHAT = '''@UTF8
@Begin
@Languages:\tdeu
@Participants:\tPAR Subject, INV Investigator
@ID:\tdeu|PARLO|PAR|65;|female|{label}||Subject|12||
@Media:\t{stem}, audio
*INV:\tBitte erzählen . \x150_100\x15
*PAR:\tÄhm ich [/] ich
\tkomme . \x15100_900\x15
%wor:\tÄhm \x15100_200\x15 ich \x15200_300\x15 ich
\t\x15300_400\x15 komme \x15400_900\x15 .
%com:\tA preserved comment
\twith a continuation
@End
'''


@pytest.fixture
def corpus_root(tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    (root / '0metadata.cdc').write_text('@UTF8\nTitle:\tSynthetic corpus\n')
    rows = []
    for group, label in [('hc', 'control'), ('MCI', 'patient_mci'), ('Ad', 'patient_ad')]:
        folder = root / group / 'nested'
        folder.mkdir(parents=True)
        for i in range(6):
            pid = f'{group.upper()}{i}'
            rows.append({'patient_id': pid, 'type': label, 'mmse_value': '' if i == 0 else str(20 + i),
                         'cdr_value': '0,5', 'age': '65', 'custom': '001'})
            for task in ['reading', 'recall']:
                if i == 0 and task == 'recall':
                    continue
                stem = f'{pid}_{task}'
                signal = np.column_stack([np.linspace(-.5, .5, 8000), np.full(8000, .25)])
                sf.write(folder / f'{stem}.WAV', signal, 8000)
                (folder / f'{stem}.CHA').write_text(CHAT.format(label=label, stem=stem))
    with (root / '0metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root
