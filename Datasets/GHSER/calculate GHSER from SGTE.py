##本代码运行需要pycalphad环境


from pycalphad import Database
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# 定义符号变量
T = sp.Symbol('T')

# 加载 TDB 文件
tdb_file = "纯相金属GHSER数据(SGTE).tdb"
tdb = Database(tdb_file)

# 获取所有 GHSER 公式
ghser_formulas = {}
for symbol_name, expression in tdb.symbols.items():
    if symbol_name.startswith('GHSER'):
        ghser_formulas[symbol_name] = str(expression)

# 手动定义需要重新计算的化学式列表
compounds_to_recalculate = [
'S',
'Co',
'Cu',
'Sb',
'Rb'




    #'Pd','Ba','In','Rh','Mg','Ga'
    #'Sb','Se','Ni','Pd','Pt','Ru','Zn',#测试
#'ZnPt','Zn3Pt','Ni3Zn','NiZn','Pd3Zn','Pd2Zn','PdZn','PdZn2','RuZn3',#用于预测的九个金属间化合物相计算GHSER来计算理论DFT计算的生成吉布斯自由能

#公式识别错误，修改需要减去GHSER的金属间化合物
# 'Yb11Sb10',
# 'InPd2',
# 'Ga2Te3',
# 'Bi2Te',
# 'Yb5Sb3',
# 'Mg2Ni',
# 'In3Pd5',
# 'Ga2Te5',
# 'Bi4Te3',
# 'Zr3Fe',
# 'In3Pd2',
# 'GaTe',
# 'In7Pd3',
# 'YbSb2',
# 'Mg2Ni','Zr3Fe','Yb2Ni17',
# 'Fe3Lu',
# 'YbNi3',
# 'YbNi',
# 'Fe23Lu6',
# 'Cu2Sb',
# 'YbNi2',
# 'Cu10Sb3',
# 'Cu3Sb',
# 'Fe2Lu',
# 'YbNi5',
]

# 解析化学式的函数（优化以处理复杂系数）
def parse_formula(formula):
    # 初始化结果字典
    result = {'elem1': None, 'elem2': None, 'coeff1': 1, 'coeff2': 0}
    
    # 使用正则表达式分解化学式
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    
    if not elements:
        return result  # 如果解析失败，返回默认值
    
    # 处理元素和系数
    for i, (elem, coeff) in enumerate(elements):
        elem_with_ghser = f'GHSER{elem.upper()}'
        if i == 0:
            result['elem1'] = elem_with_ghser
            result['coeff1'] = int(coeff) if coeff else 1
        else:
            result['elem2'] = elem_with_ghser
            result['coeff2'] = int(coeff) if coeff else 1
    
    return result

# 温度范围
T_range = np.arange(300, 1001, 100)#按步长生成列表

# 存储结果
results_dict = {}

# 处理每个化合物
for formula in compounds_to_recalculate:
    combo = parse_formula(formula)
    elem1 = combo['elem1']
    elem2 = combo['elem2']
    coeff1 = combo['coeff1']
    coeff2 = combo['coeff2']
    
    # 生成化学式并将元素转为大写
    elem1_symbol = elem1.replace('GHSER', '').upper()
    if elem2:
        elem2_symbol = elem2.replace('GHSER', '').upper()
        chem_formula = f"{elem1_symbol}{coeff1 if coeff1 > 1 else ''}{elem2_symbol}{coeff2 if coeff2 > 1 else ''}"
        total_coeff = coeff1 + coeff2
    else:
        chem_formula = f"{elem1_symbol}{coeff1 if coeff1 > 1 else ''}"
        total_coeff = coeff1
    
    # 检查元素是否存在
    if elem1 not in ghser_formulas or (elem2 and elem2 not in ghser_formulas):
        print(f"错误：{elem1} 或 {elem2} 不在 TDB 文件中")
        continue
    
    # 解析 Piecewise 表达式
    try:
        g1_expr = sp.sympify(ghser_formulas[elem1], locals={'T': T})
        g2_expr = sp.sympify(ghser_formulas[elem2], locals={'T': T}) if elem2 else 0
    except Exception as e:
        print(f"解析失败 ({chem_formula}): {e}")
        continue
    
    # 线性组合
    a = coeff1 / total_coeff if total_coeff > 0 else 1
    b = coeff2 / total_coeff if total_coeff > 0 and elem2 else 0
    g_combined = a * g1_expr + b * g2_expr
    # 计算 Cp
    dg_dT = sp.diff(g_combined, T)
    d2g_dT2 = sp.diff(dg_dT, T)
    Cp_expr = -T * d2g_dT2
    
    Cp_values = []
    T_values = []
    GHSER=[]
    for temp in T_range:
        try:
            GHSER_val= float(g_combined.evalf(subs={T: temp}))
            Cp_val = float(Cp_expr.evalf(subs={T: temp}))
            GHSER.append(GHSER_val)
            Cp_values.append(Cp_val)
            T_values.append(temp)
        except Exception as e:
            print(f"计算失败 @ {temp:.2f} K ({chem_formula}): {e}")
            Cp_values.append(np.nan)
            T_values.append(temp)
    print(f"{formula}:GHSER={GHSER}")
    results_dict[chem_formula] = Cp_values

# 绘制曲线
plt.figure(figsize=(10, 6))
for chem_formula, Cp_values in results_dict.items():
    plt.plot(T_range, Cp_values, label=chem_formula)

plt.xlabel('Temperature (K)')
plt.ylabel('Cp_mol (J/(mol·K))')
plt.title('Heat Capacity vs Temperature for Recalculated Compounds')
plt.legend()
plt.grid(True)
plt.show()

