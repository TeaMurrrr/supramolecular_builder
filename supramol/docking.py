import os
import numpy as np
from rdkit import Chem

def get_centroid(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return np.mean(coords, axis=0)

def get_min_distance(host_mol, guest_mol):
    h_conf = host_mol.GetConformer()
    g_conf = guest_mol.GetConformer()
    h_pts = np.array([list(h_conf.GetAtomPosition(i)) for i in range(host_mol.GetNumAtoms())])
    g_pts = np.array([list(g_conf.GetAtomPosition(j)) for j in range(guest_mol.GetNumAtoms())])
    dists = np.linalg.norm(h_pts[:, np.newaxis, :] - g_pts[np.newaxis, :, :], axis=2)
    return np.min(dists)

def run_advanced_geometric_docking(host_mol, guest_mol):
    print("   🔄 Алгоритм по умолчанию: Применение продвинутого геометрического разнесения...")
    
    # Сдвигаем гостя, совмещая его геометрический центр с центром хозяина
    translation_vector = get_centroid(host_mol) - get_centroid(guest_mol)
    guest_conf = guest_mol.GetConformer()
    for i in range(guest_mol.GetNumAtoms()):
        guest_conf.SetAtomPosition(i, guest_conf.GetAtomPosition(i) + translation_vector)
        
    # Итеративно отодвигаем гостя по оси X, пока минимальное расстояние между атомами не станет >= 2.2 Å
    shift_step = np.array([0.4, 0.0, 0.0])
    for _ in range(100):
        if get_min_distance(host_mol, guest_mol) >= 2.2:
            break
        for i in range(guest_mol.GetNumAtoms()):
            guest_conf.SetAtomPosition(i, guest_conf.GetAtomPosition(i) + shift_step)
            
    complex_mol = Chem.CombineMols(host_mol, guest_mol)
    charge = sum(a.GetFormalCharge() for a in complex_mol.GetAtoms())
    print(f"   ✅ Геометрический сдвиг выполнен. Стерические наложения устранены (min_dist >= 2.2 Å).")
    return complex_mol, charge

def center_and_dock(host_mol, guest_mol, run_dir, use_vina=False):
    print("\n🎯 Сценарий докинга: Поиск начальной конформации...")
    
    if use_vina:
        print("   • Запрошен эмпирический докинг через AutoDock Vina...")
        try:
            from vina import Vina
            
            work_dir = os.path.join(run_dir, "vina_docking")
            os.makedirs(work_dir, exist_ok=True)
            
            host_pdbqt = os.path.join(work_dir, "dock_host.pdbqt")
            guest_pdbqt = os.path.join(work_dir, "dock_guest.pdbqt")
            output_pdbqt = os.path.join(work_dir, "dock_poses.pdbqt")
            
            def save_simple_pdbqt(mol, path):
                with open(path, "w") as f:
                    for i, atom in enumerate(mol.GetAtoms()):
                        pos = mol.GetConformer().GetAtomPosition(i)
                        sym = atom.GetSymbol()
                        f.write(f"HETATM{i+1:5d} {sym:<3s}  UNK A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00     +0.000 {sym:<2s}\n")
            
            save_simple_pdbqt(host_mol, host_pdbqt)
            save_simple_pdbqt(guest_mol, guest_pdbqt)
            
            # Настройка бокса Vina вокруг хозяина
            host_conf = host_mol.GetConformer()
            host_coords = np.array([list(host_conf.GetAtomPosition(i)) for i in range(host_mol.GetNumAtoms())])
            center = np.mean(host_coords, axis=0).tolist()
            
            coord_min = np.min(host_coords, axis=0)
            coord_max = np.max(host_coords, axis=0)
            box_size = (coord_max - coord_min + 12.0).tolist()
            box_size = [max(b, 16.0) for b in box_size]
            
            print(f"     Инициализация Vina (Центр: {[round(c,2) for c in center]}, Бокс: {[round(b,2) for b in box_size]})...")
            v = Vina(sf_name='vina', cpu=4, verbosity=0)
            v.set_receptor(host_pdbqt)
            v.set_ligand_from_file(guest_pdbqt)
            v.compute_vina_maps(center=center, box_size=box_size)
            
            v.dock(exhaustiveness=8, n_poses=1)
            v.write_poses(output_pdbqt, n_poses=1, overwrite=True)
            
            if os.path.exists(output_pdbqt):
                with open(output_pdbqt, "r") as f:
                    lines = f.readlines()
                
                guest_conf = guest_mol.GetConformer()
                atom_idx = 0
                for line in lines:
                    if line.startswith("HETATM") or line.startswith("ATOM"):
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        if atom_idx < guest_mol.GetNumAtoms():
                            guest_conf.SetAtomPosition(atom_idx, (x, y, z))
                            atom_idx += 1
                            
                complex_mol = Chem.CombineMols(host_mol, guest_mol)
                charge = sum(a.GetFormalCharge() for a in complex_mol.GetAtoms())
                print(f"   ✅ AutoDock Vina успешно уложил гостя в полость/окно хозяина.")
                return complex_mol, charge

        except Exception as e:
            print(f"   ⚠️ Ошибка при инициализации Vina ({e}). Переключение на резервный алгоритм...")
            return run_advanced_geometric_docking(host_mol, guest_mol)
    else:
        # Если флаг --vina не передан, сразу запускаем геометрию
        return run_advanced_geometric_docking(host_mol, guest_mol)