import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_shanxi_pension():
    # 基础养老保险参数
    base_params = {
        "name": "基础养老保险",
        "start_age": 60,       # 开始领取年龄
        "base_pension": 123,   # 基础养老金标准(元/月)
        "n_years_pay": 15,     # 缴费年限
        "annual_rate": 0.03,   # 个人账户年化收益率(3%)
        "months_divide": 139,  # 计发月数(60岁退休)
        "levels": {            # 缴费档次及政府补贴
            200: 35, 300: 40, 500: 60, 700: 80, 1000: 100,
            1500: 140, 2000: 180, 3000: 220, 4000: 260, 5000: 300
        }
    }
    
    # 补充养老保险参数
    extra_params = {
        "name": "补充养老保险",
        "start_age": 65,       # 开始领取年龄
        "base_pension": 20,    # 补充基础养老金标准(元/月)
        "n_years_pay": 15,     # 缴费年限
        "annual_rate": 0.03,   # 个人账户年化收益率(3%)
        "months_divide": 101,  # 计发月数(65岁退休)
        "levels": {            # 缴费档次及政府补贴
            200: 70, 500: 120, 1000: 200, 2000: 360, 5000: 600
        },
        "seniority_bonus": 15  # 缴费年限养老金(15年)
    }
    
    # 接收年限范围(1-30年)
    receive_years = list(range(1, 31))
    
    print("山西省城乡居民养老保险收益率分析")
    print(f"缴费年限: {base_params['n_years_pay']}年, 账户收益率: {base_params['annual_rate']*100}%")
    print("=" * 100)
    
    # 分别计算基础养老金和补充养老金的收益率
    def calculate_single_pension(params):
        results = []
        
        for pay, subsidy in params["levels"].items():
            # 计算个人账户积累总额
            total_payment = pay + subsidy
            account_value = total_payment * ((1 + params["annual_rate"]) ** params["n_years_pay"] - 1) / params["annual_rate"]
            
            # 计算个人账户养老金(月)
            personal_pension = account_value / params["months_divide"]
            
            # 计算月总养老金
            if "seniority_bonus" in params:
                total_monthly_pension = params["base_pension"] + params["seniority_bonus"] + personal_pension
            else:
                total_monthly_pension = params["base_pension"] + personal_pension
            
            # 构建现金流
            cash_flows = []
            
            # 缴费期
            for _ in range(params["n_years_pay"]):
                cash_flows.append(-pay)
            
            # 计算不同领取年限的IRR
            irr_results = []
            for years in receive_years:
                temp_flows = cash_flows.copy()
                
                # 添加领取期现金流
                for year in range(years):
                    # 考虑开始领取年龄的延迟
                    if year < (params["start_age"] - 60):
                        # 尚未开始领取
                        temp_flows.append(0)
                    else:
                        # 开始领取养老金
                        temp_flows.append(total_monthly_pension * 12)
                
                # 计算IRR
                try:
                    irr = npf.irr(temp_flows)
                    irr_percent = round(irr * 100, 2) if irr is not None else None
                except:
                    irr_percent = None
                
                irr_results.append(irr_percent)
            
            # 保存结果
            results.append({
                "缴费档次": pay,
                "政府补贴": subsidy,
                "月养老金": round(total_monthly_pension, 1),
                "收益率": irr_results
            })
        
        return pd.DataFrame(results), params["name"]
    
    # 计算合并养老金收益率
    def calculate_combined_pension():
        results = []
        
        for base_pay, base_subsidy in base_params["levels"].items():
            # 计算基础养老保险
            base_total = base_pay + base_subsidy
            base_account = base_total * ((1 + base_params["annual_rate"]) ** base_params["n_years_pay"] - 1) / base_params["annual_rate"]
            base_monthly = base_account / base_params["months_divide"]
            base_total_monthly = base_params["base_pension"] + base_monthly
            
            for extra_pay, extra_subsidy in extra_params["levels"].items():
                # 计算补充养老保险
                extra_total = extra_pay + extra_subsidy
                extra_account = extra_total * ((1 + extra_params["annual_rate"]) ** extra_params["n_years_pay"] - 1) / extra_params["annual_rate"]
                extra_monthly = extra_account / extra_params["months_divide"]
                extra_total_monthly = extra_params["base_pension"] + extra_params["seniority_bonus"] + extra_monthly
                
                # 合并养老金
                total_monthly_60_64 = base_total_monthly  # 60-64岁只领基础养老
                total_monthly_65plus = base_total_monthly + extra_total_monthly  # 65岁起领两份
                
                # 构建现金流
                cash_flows = []
                
                # 缴费期
                for _ in range(base_params["n_years_pay"]):
                    cash_flows.append(-(base_pay + extra_pay))
                
                # 计算不同领取年限的IRR
                irr_results = []
                for years in receive_years:
                    temp_flows = cash_flows.copy()
                    
                    # 计算领取期
                    for year in range(1, years + 1):
                        age = 60 + year - 1  # 当前年龄
                        
                        if age < 65:
                            # 60-64岁只领取基础养老金
                            temp_flows.append(total_monthly_60_64 * 12)
                        else:
                            # 65岁起领取基础+补充
                            temp_flows.append(total_monthly_65plus * 12)
                    
                    # 计算IRR
                    try:
                        irr = npf.irr(temp_flows)
                        irr_percent = round(irr * 100, 2) if irr is not None else None
                    except:
                        irr_percent = None
                    
                    irr_results.append(irr_percent)
                
                # 保存结果
                results.append({
                    "基础缴费": base_pay,
                    "基础补贴": base_subsidy,
                    "补充缴费": extra_pay,
                    "补充补贴": extra_subsidy,
                    "60-64岁月领": round(total_monthly_60_64, 1),
                    "65+岁月领": round(total_monthly_65plus, 1),
                    "收益率": irr_results
                })
        
        return pd.DataFrame(results)
    
    # 计算并显示结果
    base_df, base_name = calculate_single_pension(base_params)
    extra_df, extra_name = calculate_single_pension(extra_params)
    combined_df = calculate_combined_pension()
    
    # 输出基础养老金结果
    print(f"\n{base_name}收益率分析:")
    print(base_df[["缴费档次", "政府补贴", "月养老金"]].to_string(index=False))
    
    print("\n不同领取年限收益率(%):")
    print("缴费档次", end=" ")
    for year in receive_years:
        print(f"{year}年", end=" ")
    print()
    
    for _, row in base_df.iterrows():
        print(f"{row['缴费档次']:6}", end=" ")
        for irr in row["收益率"]:
            print(f"{irr:5.1f}" if irr is not None else "  -- ", end=" ")
        print()
    
    # 输出补充养老金结果
    print(f"\n{extra_name}收益率分析:")
    print(extra_df[["缴费档次", "政府补贴", "月养老金"]].to_string(index=False))
    
    print("\n不同领取年限收益率(%):")
    print("缴费档次", end=" ")
    for year in receive_years:
        print(f"{year}年", end=" ")
    print()
    
    for _, row in extra_df.iterrows():
        print(f"{row['缴费档次']:6}", end=" ")
        for irr in row["收益率"]:
            print(f"{irr:5.1f}" if irr is not None else "  -- ", end=" ")
        print()
    
    # 输出合并养老金结果
    print("\n基础+补充养老保险合并收益率分析:")
    print(combined_df[["基础缴费", "基础补贴", "补充缴费", "补充补贴", "60-64岁月领", "65+岁月领"]].head().to_string(index=False))
    print("... (更多组合请查看完整数据)")
    
    print("\n不同领取年限合并收益率(%):")
    print("基础缴费 | 补充缴费", end=" ")
    for year in receive_years:
        print(f"{year}年", end=" ")
    print()
    
    # 只显示部分组合示例
    sample_combinations = [
        (200, 200), (500, 500), (1000, 1000), (2000, 2000), (5000, 5000)
    ]
    
    for _, row in combined_df.iterrows():
        if (row["基础缴费"], row["补充缴费"]) in sample_combinations:
            print(f"{row['基础缴费']:6} | {row['补充缴费']:6}", end=" ")
            for irr in row["收益率"]:
                print(f"{irr:5.1f}" if irr is not None else "  -- ", end=" ")
            print()
    
    # 可视化部分结果
    visualize_results(base_df, extra_df, combined_df, base_name, extra_name)
    
    return base_df, extra_df, combined_df

def visualize_results(base_df, extra_df, combined_df, base_name, extra_name):
    """可视化关键结果"""
    plt.figure(figsize=(15, 10))
    
    # 基础养老金收益率曲线
    plt.subplot(2, 2, 1)
    for _, row in base_df.iterrows():
        pay = row["缴费档次"]
        irrs = [irr for irr in row["收益率"] if irr is not None]
        plt.plot(range(1, len(irrs)+1), irrs, label=f"{pay}元")
    
    plt.title(f"{base_name}不同缴费档次收益率")
    plt.xlabel("领取年限(年)")
    plt.ylabel("收益率(%)")
    plt.grid(True)
    plt.legend(title="缴费档次", loc="lower right")
    
    # 补充养老金收益率曲线
    plt.subplot(2, 2, 2)
    for _, row in extra_df.iterrows():
        pay = row["缴费档次"]
        irrs = [irr for irr in row["收益率"] if irr is not None]
        plt.plot(range(1, len(irrs)+1), irrs, label=f"{pay}元")
    
    plt.title(f"{extra_name}不同缴费档次收益率")
    plt.xlabel("领取年限(年)")
    plt.ylabel("收益率(%)")
    plt.grid(True)
    plt.legend(title="缴费档次", loc="lower right")
    
    # 合并养老金收益率曲线(示例)
    plt.subplot(2, 1, 2)
    sample_combinations = [(500, 500), (1000, 1000), (2000, 2000)]
    
    for _, row in combined_df.iterrows():
        base_pay = row["基础缴费"]
        extra_pay = row["补充缴费"]
        if (base_pay, extra_pay) in sample_combinations:
            irrs = [irr for irr in row["收益率"] if irr is not None]
            plt.plot(range(1, len(irrs)+1), irrs, label=f"基础{base_pay}+补充{extra_pay}")
    
    plt.title("合并养老保险收益率(示例组合)")
    plt.xlabel("领取年限(年)")
    plt.ylabel("收益率(%)")
    plt.grid(True)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("shanxi_pension_analysis.png")
    print("\n图表已保存为 shanxi_pension_analysis.png")

if __name__ == "__main__":
    # 安装必要库: pip install numpy-financial pandas matplotlib
    base_df, extra_df, combined_df = calculate_shanxi_pension()
    
    print("\n分析说明:")
    print("1. 基础养老金从60岁开始领取，补充养老金从65岁开始领取")
    print("2. 合并收益率计算时，60-64岁只领取基础养老金，65岁起领取两份养老金")
    print("3. 收益率基于缴费15年、年化收益3%计算")
    print("4. 基础养老金计发月数139个月(60岁退休标准)")
    print("5. 补充养老金计发月数101个月(65岁退休标准)")
    print("6. 图表已保存为 shanxi_pension_analysis.png")