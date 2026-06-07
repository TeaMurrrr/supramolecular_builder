import os
import numpy as np
from rdkit import Chem

def print_log_tail(filepath, n=25):
    """Выводит финальные строки лога при падении внешних утилит."""
    print(f"\n🛑 [АНАЛИЗ ОШИБКИ] Последние {n} строк из файла: {os.path.basename(filepath)}")
    print("=" * 70)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                print(f"   | {line.strip()}")
    else:
        print("   ❌ Файл лога отсутствует.")
    print("=" * 70)

def create_xtb_input(directory):
    """Генерирует конфигурационный файл конфигурации SCC для xTB."""
    inp_path = os.path.join(directory, "xtb.inp")
    with open(inp_path, "w") as f:
        f.write("$scc\n   maxcycle=600\n   temp=400\n$end\n")
    return "xtb.inp"

def fix_formal_charges(mol):
    """Корректирует формальные заряды для стандартных валентных состояний N и O."""
    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        total_valence = atom.GetTotalValence()
        
        if symbol == "N" and total_valence == 4 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(1)
        elif symbol == "O" and total_valence == 1 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(-1)
        elif symbol == "O" and total_valence == 3 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(1)

    try:
        Chem.SanitizeMol(mol)
    except:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    return mol

def update_mol_from_xyz(mol, xyz_path):
    """Обновляет 3D координаты конформера RDKit на основе XYZ файла."""
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        parts = lines[i + 2].split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        conf.SetAtomPosition(i, (x, y, z))
    return mol

def get_centroid(mol):
    """Вычисляет геометрический центр масс молекулы."""
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return np.mean(coords, axis=0)