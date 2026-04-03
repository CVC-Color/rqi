import os
import torch
import csv
from rqi import RQI
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
# import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run():
    rqimodel = RQI(pretrained=True).cuda()

    image_root = 'your_path/SRIQA-Bench/images'
    csv_root = 'your_path/SRIQA-Bench/SRIQA.csv'
    output_csv = "RQI_results_SRIQA.csv"
    
    with open(csv_root, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        model_names = fieldnames[1:]

        results = []
        for row in reader:
            name = row["name"]
            print(f"Processing {name}...")

            row_result = {"name": name}
            gt_path = os.path.join(image_root, "Original/", f"{name}_Original.png")

            for model in model_names:
                image_path = os.path.join(image_root, model, f"{name}_{model}.png")

                if os.path.exists(image_path):
                    score = rqimodel(image_path, gt_path)
                else:
                    print(f"Missing {image_path}")
                    score = None

                row_result[model] = score

            results.append(row_result)
            print(results)

    output_fields = ["name"] + model_names
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Evaluation done, results in: {output_csv}")
    

    # ========= Calculating SRCC, PLCC ==========
    file1 = csv_root
    file2 = 'RQI_results_SRIQA.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    cols = [c for c in df1.columns if c != 'name']
    assert cols == [c for c in df2.columns if c != 'name'], "列名不一致！"

    row_srcc, row_plcc = [], []

    # ===== Calculate SRCC / PLCC for each source image =====
    for i in range(len(df1)):
        x = df1.loc[i, cols].values
        y = df2.loc[i, cols].values

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        srcc, _ = spearmanr(x, y)
        plcc, _ = pearsonr(x, y)

        row_srcc.append(srcc)
        row_plcc.append(plcc)

    mean_srcc = np.mean(row_srcc)
    mean_plcc = np.mean(row_plcc)

    print(f"SRCC mean: {mean_srcc:.4f}")
    print(f"PLCC mean: {mean_plcc:.4f}")

    # ===== Calculate overall SRCC / PLCC =====
    x_all = df1[cols].values.flatten()
    y_all = df2[cols].values.flatten()

    srcc_all, _ = spearmanr(x_all, y_all)
    plcc_all, _ = pearsonr(x_all, y_all)

    print(f"Overall SRCC: {srcc_all:.4f}")
    print(f"Overall PLCC: {plcc_all:.4f}")


if __name__ == "__main__":
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    run()