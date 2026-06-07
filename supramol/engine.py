import os
import subprocess
from rdkit import Chem
from .utils import create_xtb_input, print_log_tail

def fast_quantum_docking(complex_mol, charge, run_dir, gfn_level, mdlen, mode, threads, noreftopo=False):
    work_dir = os.path.join(run_dir, "quantum_run")
    os.makedirs(work_dir, exist_ok=True)
    
    num_atoms = complex_mol.GetNumAtoms()
    is_complex = len(Chem.GetMolFrags(complex_mol)) > 1
    method_title = "GFN-FF (Силовое поле)" if gfn_level == "ff" else f"GFN{gfn_level}-xTB"
    
    print(f"\n📊 ХАРАКТЕРИСТИКИ СИСТЕМЫ И НАСТРОЙКИ КВАНТОВОГО ДВИЖКА:")
    print(f"   • Число атомов:             {num_atoms}\n   • Заряд системы:            {charge}")
    print(f"   • Уровень теории xTB/CREST:  {method_title}\n   • Выделено потоков (cores): {threads}")
    print(f"   • Скоростной режим CREST:   {mode.upper()}")
    print(f"   • Тип topology:             {'КОМПЛЕКС (--nci)' if is_complex else 'ОДИНОЧНАЯ МОЛЕКУЛА'}")
    
    input_xyz = os.path.join(work_dir, "raw_complex.xyz")
    Chem.MolToXYZFile(complex_mol, input_xyz)
    inp_file = create_xtb_input(work_dir)
    
    # --- ШАГ 1: Loose релаксация комплекса ---
    print(f"\n🛠️ [Шаг 1/3] Запуск релаксации комплекса (xTB)...")
    log_step1 = os.path.join(work_dir, "step1_preopt.log")
    opt_flag = "--gfnff" if gfn_level == "ff" else f"--gfn {gfn_level}"
    preopt_cmd = f"xtb raw_complex.xyz --opt loose {opt_flag} --chrg {charge} --input {inp_file}"
    
    with open(log_step1, "w") as f:
        res = subprocess.run(preopt_cmd, cwd=work_dir, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
    target_xyz = os.path.join(work_dir, "xtbopt.xyz")
    if res.returncode != 0 or not os.path.exists(target_xyz):
        print("   ⚠️ xTB выдал ошибку на Шаге 1. Используем сырые начальные координаты.")
        target_xyz = input_xyz

    # --- ШАГ 2: Конформационный поиск (CREST) ---
    print(f"\n🧬 [Шаг 2/3] Запуск конформационного поиска CREST...")
    log_step2 = os.path.join(work_dir, "step2_crest.log")
    
    crest_args = [
        "crest", os.path.basename(target_xyz),
        "--chrg", str(charge), "-T", str(threads),
        "--norotamer", "--xopt", inp_file
    ]
    crest_args.append("--gfnff" if gfn_level == "ff" else f"--gfn {gfn_level}")
    if mode in ["quick", "mquick"]: crest_args.append(f"--{mode}")
    if mdlen is not None: crest_args.extend(["--mdlen", str(mdlen)]) 
    else: crest_args.extend(["--mdlen", str(10)])
    if is_complex: crest_args.append("--nci")
    if noreftopo: crest_args.append("--noreftopo")
        
    with open(log_step2, "w") as f:
        res = subprocess.run(" ".join(crest_args), cwd=work_dir, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
    crest_result = os.path.join(work_dir, "crest_best.xyz")
    if res.returncode != 0 or not os.path.exists(crest_result):
        print("   ❌ Ошибка: Выполнение CREST завершилось аварийно!")
        print_log_tail(log_step2)
        return None

    # --- ШАГ 3: Финальная оптимизация структуры ---
    log_step3 = os.path.join(work_dir, "step3_final.log")
    final_gfn = gfn_level if gfn_level in ["ff", "0"] else "2"
    print(f"\n⚡ [Шаг 3/3] Финальная квантовая полировка (xTB GFN{final_gfn})...")
    final_opt_flag = "--gfnff" if final_gfn == "ff" else f"--gfn {final_gfn}"
    final_xtb_cmd = f"xtb crest_best.xyz --opt loose {final_opt_flag} --chrg {charge} --input {inp_file}"
    
    with open(log_step3, "w") as f:
        res = subprocess.run(final_xtb_cmd, cwd=work_dir, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
    final_xyz = os.path.join(work_dir, "xtbopt.xyz")
    if res.returncode != 0 or not os.path.exists(final_xyz):
        print("   ⚠️ Финальная релаксация не сошлась. Берём геометрию CREST.")
        return crest_result
        
    return final_xyz