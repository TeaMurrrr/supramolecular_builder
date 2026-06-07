import os
import sys
import argparse
from rdkit import Chem

from supramol.config import init_environment, patch_gfn0_path
from supramol.loader import load_molecule
from supramol.docking import center_and_dock
from supramol.engine import fast_quantum_docking
from supramol.utils import update_mol_from_xyz

def main():
    parser = argparse.ArgumentParser(description="Супрамолекулярный конвейер (Модульная версия).")
    parser.add_argument("-host", type=str, help="Файл или SMILES Хозяина")
    parser.add_argument("-guest", type=str, help="Файл или SMILES Гостя")
    parser.add_argument("-i", type=str, help="Входной файл одиночной структуры")
    parser.add_argument("-o", type=str, help="Имя выходного проекта")
    
    parser.add_argument("--chrg", type=int, default=None, help="Вручную задать формальный заряд всей системы")
    parser.add_argument("--vina", action="store_true", help="Включить предварительный докинг через AutoDock Vina (вместо геометрии)")
    
    parser.add_argument("--gfn", type=str, choices=["0", "1", "2", "ff"], default="1", help="Метод оптимизации")
    parser.add_argument("--mdlen", type=float, default=None)
    parser.add_argument("--mode", type=str, choices=["standard", "quick", "mquick"], default="standard")
    parser.add_argument("-t", "--threads", type=int, default=6)
    parser.add_argument("--rebuild", action="store_true", help="Принудительный пересбор 3D RDKit")
    parser.add_argument("--noreftopo", action="store_true", help="Отключить проверку изменения топологии связей")
    
    args = parser.parse_args()

    # Среда и конфигурация
    init_environment(threads=args.threads)
    if args.gfn == "0":
        patch_gfn0_path()

    if not args.i and not args.host:
        print("❌ Критическая ошибка: Не указан источник данных.")
        sys.exit(1)

    # Формируем имя проекта для создания единой папки запуска
    if args.o:
        project_name = args.o.replace(".sdf", "")
    elif args.i:
        project_name = os.path.splitext(os.path.basename(args.i))[0] if os.path.exists(args.i) else "isolated_mol"
    else:
        h_name = os.path.splitext(os.path.basename(args.host))[0] if os.path.exists(args.host) else "host"
        if args.guest:
            g_name = os.path.splitext(os.path.basename(args.guest))[0] if os.path.exists(args.guest) else "guest"
            project_name = f"{h_name}_{g_name}"
        else:
            project_name = f"opt_{h_name}"

    # Создаем общую папку для всех временных файлов этого запуска
    run_dir = os.path.join("run", project_name)
    os.makedirs(run_dir, exist_ok=True)

    print("======================================================================")
    print("🚀 НАЧАЛО РАБОТЫ ОБНОВЛЕННОГО СУПРАМОЛЕКУЛЯРНОГО КОНВЕЙЕРА")
    print(f"📂 Все временные файлы сохраняются в: {run_dir}")
    print("======================================================================")

    # Логика сборки/загрузки
    if args.i:
        complex_mol, _ = load_molecule(args.i, "Изолированный ввод", run_dir, rebuild=args.rebuild, gfn_level=args.gfn)
        charge = args.chrg if args.chrg is not None else sum(a.GetFormalCharge() for a in complex_mol.GetAtoms())
    else:
        host_mol, _ = load_molecule(args.host, "Host", run_dir, rebuild=args.rebuild, gfn_level=args.gfn)
        if args.guest:
            guest_mol, _ = load_molecule(args.guest, "Guest", run_dir, rebuild=args.rebuild, gfn_level=args.gfn)
            
            complex_mol, auto_charge = center_and_dock(host_mol, guest_mol, run_dir, use_vina=args.vina)
            charge = args.chrg if args.chrg is not None else auto_charge
        else:
            complex_mol = host_mol
            charge = args.chrg if args.chrg is not None else sum(a.GetFormalCharge() for a in complex_mol.GetAtoms())

    # Расчет ядра
    best_xyz_path = fast_quantum_docking(
        complex_mol, charge, run_dir, 
        gfn_level=args.gfn, mdlen=args.mdlen, mode=args.mode, threads=args.threads,
        noreftopo=args.noreftopo
    )
    
    # Финализация и сохранение результатов
    if best_xyz_path:
        print("\n🧬 [Интеграция] Перенос квантовых XYZ координат на исходный граф RDKit...")
        final_complex = update_mol_from_xyz(complex_mol, best_xyz_path)
        
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{project_name}_final.sdf")
        
        Chem.MolToMolFile(final_complex, output_file)
        
        print("=" * 70)
        print(f"🎉 УСПЕШНОЕ ЗАВЕРШЕНИЕ РАСЧЕТА!\n📦 Итоговый 3D SDF файл: {output_file}")
        print("=" * 70)
    else:
        print("\n❌ Расчет завершился неудачно.")
        sys.exit(1)

if __name__ == "__main__":
    main()