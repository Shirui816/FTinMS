import numpy as np





def grab_iter_dual(i, bond_hash, mol_used, body_hash=None):
    s = [i]
    r = []
    while s:
        v = s.pop()
        if not mol_used[v]:
            r.append(v)
            mol_used[v] = True
            # for w in bond_hash[v]:
            # s.append(w)
            s.extend(bond_hash[v])
            if not body_hash:
                continue
            for w in body_hash.get(v):
                s.append(w)
                for x in bond_hash[w]:
                    s.append(x)
    return r

def bond_hash_dualdirect(bond, natoms):
    """
    :param bond: bond data in hoomdxml format (name, id1, id2)
    :param natoms: total number of particles
    :return: hash table of with value in {bondname1: [idxes], bondname2:[idxes]...} for each particle (in dual direct)
    """
    bond_hash_nn = {}
    print('Building bond hash...')
    if not isinstance(bond, np.ndarray):
        return {}
    for i in range(natoms):
        bond_hash_nn[i] = []
    for b in bond:
        idx = b[1]
        jdx = b[2]
        bond_hash_nn[idx].append(jdx)
        bond_hash_nn[jdx].append(idx)
    print('Done.')
    return bond_hash_nn

def molecules(bond, natoms):
    bond_hash = bond_hash_dualdirect(bond, natoms)
    mol_used = {}
    for i in range(natoms):
        mol_used[i] = False
    _ret, ml = [], []
    for i in range(natoms):
        mol = grab_iter_dual(i, bond_hash, mol_used)
        if len(mol) > 1:
            _ret.append(mol)
            ml.append(len(mol))
    ret = np.zeros((len(_ret), max(ml)), dtype=np.int64) - 1
    for i, mol in enumerate(_ret):
        ret[i][:ml[i]] = _ret[i]
    return ret, np.array(ml, dtype=np.int64), bond_hash
