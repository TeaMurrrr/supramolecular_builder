import os
import sys
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from .utils import fix_formal_charges, create_xtb_input, update_mol_from_xyz

def load_molecule(input_source, label, run_dir, rebuild=False, gfn_level="1"):
    print(f"📖 Загрузка структуры [{label}] из источника: {input_source}")
    if not input_source:
        return None, "empty"

    mol = None
    is_sdf = False

    if os.path.exists(input_source) and input_source.lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(input_source, removeHs=False, sanitize=False)
        mol = next(suppl)
        name = mol.GetProp("_Name").strip() if mol and mol.HasProp("_Name") else os.path.splitext(os.path.basename(input_source))[0]
        is_sdf = True
    else:
        try:
            mol = Chem.MolFromSmiles(input_source, sanitize=False)
            name = "smiles_mol"
        except: pass

    if mol is None:
        print(f"❌ Ошибка: Не удалось прочитать структуру {input_source}")
        sys.exit(1)

    mol = fix_formal_charges(mol)
    force_rebuild = rebuild or (not is_sdf) or (mol.GetNumConformers() == 0)

    if not force_rebuild:
        print(f"   • [Режим: Готорая геометрия] Обнаружены 3D-координаты. Заряды проверены.")
        mol_with_hs = mol
        
        if mol_with_hs.GetNumAtoms() > 1:
            method_label = "GFN-FF" if gfn_level == "ff" else f"GFN{gfn_level}"
            print(f"     ⚡ [xTB Изолированный] Предварительное расслабление структуры [{label}] ({method_label})...")
            mol_charge = sum(a.GetFormalCharge() for a in mol_with_hs.GetAtoms())
            
            tmp_dir = os.path.join(run_dir, f"preopt_{label}_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_xyz = os.path.join(tmp_dir, "isolated_input.xyz")
            Chem.MolToXYZFile(mol_with_hs, tmp_xyz)
            
            inp_file = create_xtb_input(tmp_dir)
            opt_flag = "--gfnff" if gfn_level == "ff" else f"--gfn {gfn_level}"
            xtb_cmd = f"xtb isolated_input.xyz --opt loose {opt_flag} --chrg {mol_charge} --input {inp_file}"
            
            with open(os.path.join(tmp_dir, "preopt_xtb.log"), "w") as f_log:
                res = subprocess.run(xtb_cmd, cwd=tmp_dir, shell=True, stdout=f_log, stderr=subprocess.STDOUT)
                
            opt_xyz = os.path.join(tmp_dir, "xtbopt.xyz")
            if res.returncode == 0 and os.path.exists(opt_xyz):
                mol_with_hs = update_mol_from_xyz(mol_with_hs, opt_xyz)
                print(f"     ✅ [xTB Изолированный] Геометрия успешно оптимизирована.")
            else:
                print(f"     ⚠️ [xTB Изолированный] Сбой оптимизации. Оставляем исходную геометрию.")
        else:
            print("     ✅ Структура состоит из 1 атома, пропускаем pre-opt.")
    else:
        print(f"   • [Режим: REBUILD] Запуск полного цикла построения 3D-координат...")
        mol_with_hs = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.clearConfs = True
        embed_status = AllChem.EmbedMolecule(mol_with_hs, params)
        
        if embed_status == -1:
            params_v2 = AllChem.ETKDGv2()
            params_v2.randomSeed = 42
            params_v2.clearConfs = True
            embed_status = AllChem.EmbedMolecule(mol_with_hs, params_v2)
            
        if embed_status == -1:
            params.useRandomCoords = True
            params.maxIterations = 5000
            embed_status = AllChem.EmbedMolecule(mol_with_hs, params)

        if embed_status == -1:
            print("     ❌ Критическая ошибка: RDKit не смог построить конформер.")
            sys.exit(1)
            
        if AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=3500, mmffVariant='MMFF94') != 0:
            AllChem.UFFOptimizeMolecule(mol_with_hs, maxIters=4500)
            
        method_label = "GFN-FF" if gfn_level == "ff" else f"GFN{gfn_level}"
        print(f"     ⚡ [xTB Rebuild] Предварительный квантовый дожим ({method_label})...")
        mol_charge = sum(a.GetFormalCharge() for a in mol_with_hs.GetAtoms())
        
        tmp_dir = os.path.join(run_dir, f"rebuild_{label}_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_xyz = os.path.join(tmp_dir, "rdkit_output.xyz")
        Chem.MolToXYZFile(mol_with_hs, tmp_xyz)
        
        inp_file = create_xtb_input(tmp_dir)
        opt_flag = "--gfnff" if gfn_level == "ff" else f"--gfn {gfn_level}"
        xtb_cmd = f"xtb rdkit_output.xyz --opt loose {opt_flag} --chrg {mol_charge} --input {inp_file}"
        
        with open(os.path.join(tmp_dir, "rebuild_xtb.log"), "w") as f_log:
            res = subprocess.run(xtb_cmd, cwd=tmp_dir, shell=True, stdout=f_log, stderr=subprocess.STDOUT)
            
        opt_xyz = os.path.join(tmp_dir, "xtbopt.xyz")
        if res.returncode == 0 and os.path.exists(opt_xyz):
            mol_with_hs = update_mol_from_xyz(mol_with_hs, opt_xyz)
            print("     ✅ [xTB Rebuild] Грубая предоптимизация завершена.")
        else:
            print("     ⚠️ [xTB Rebuild] Сбой loose-оптимизации. Оставляем геометрию RDKit.")

    return mol_with_hs, name