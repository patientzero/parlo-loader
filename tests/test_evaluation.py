import json
import polars as pl
import pytest
from parlo import Corpus, SplitManifest, stratified_split, stratified_kfold


def test_splits_reproducible_isolated_across_levels(corpus_root,tmp_path):
    c=Corpus(corpus_root)
    split=c.split()
    assert split.assignments == c.split().assignments
    assert split.assignments != c.split(seed=123).assignments
    path=tmp_path/'split.json'
    split.save(path)
    assert SplitManifest.load(path).assignments == split.assignments
    sets=[set(ids) for ids in split.assignments.values()]
    assert len(set.union(*sets)) == 18
    assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
    for query in [c.recordings,c.utterances,c.words,c.filter(task='recall').recordings]:
        expanded=[set(split.apply(query,name).collect()['participant_id']) for name in split.assignments]
        assert all(not a & b for i,a in enumerate(expanded) for b in expanded[i+1:])
    for ids in split.assignments.values():
        assert {p[:2] for p in ids} == {'HC','MC','AD'}


def test_kfold_and_lopo(corpus_root):
    c=Corpus(corpus_root)
    folds=c.kfold()
    assert len(folds) == 5
    assert [f.assignments for f in folds] == [f.assignments for f in c.kfold()]
    tests=[pid for f in folds for pid in f.assignments['test']]
    assert len(tests)==len(set(tests))==18
    lopo=list(c.leave_one_out())
    assert len(lopo)==18
    assert all(len(f.assignments['test'])==1 and len(f.assignments['train'])==17 for f in lopo)


def test_infeasible_splits_and_invalid_manifest():
    query=pl.LazyFrame({'participant_id':['1','2'], 'diagnosis':['HC','HC']})
    with pytest.raises(ValueError,match='need at least 3'):
        stratified_split(query)
    with pytest.raises(ValueError,match='Reduce folds'):
        stratified_kfold(query)
    with pytest.raises(ValueError,match='summing to one'):
        stratified_split(query,proportions=(1,1,1))
    with pytest.raises(ValueError,match='leakage'):
        SplitManifest({'train':['1'],'test':['1']})
    split=SplitManifest({'train':['1'],'test':[]})
    with pytest.raises(ValueError,match='absent'):
        split.apply(query,'train')
    assert split.apply(query,'train',strict=False).collect().height == 1


def test_target_mappings_and_local_regression_exclusions(corpus_root):
    c=Corpus(corpus_root)
    multi=c.targets()
    assert multi.mapping == {'HC':0,'MCI':1,'AD':2}
    assert multi.expand().collect().height==33
    pair=c.targets(labels=['AD','HC'])
    assert pair.mapping=={'AD':0,'HC':1}
    assert pair.participants.collect().height==12
    assert pair.exclusions.collect().height==6
    assert c.targets('hc_vs_impaired').mapping=={'HC':0,'MCI':1,'AD':1}
    with pytest.warns(UserWarning,match='Excluded 3'):
        mmse=c.targets('mmse')
    assert mmse.exclusions.collect().height==3
    assert mmse.participants.collect().height==15
    assert mmse.expand('utterances').collect().height==60
    assert c.participants.collect().height==18
    assert c.targets('cdr').participants.collect()['target'].to_list()==[.5]*18
    assert mmse.split().assignments
