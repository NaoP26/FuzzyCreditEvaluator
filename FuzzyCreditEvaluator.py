import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tkinter as tk
from tkinter import messagebox

# House Evaluation
market_value = ctrl.Antecedent(np.arange(0, 1000001, 0.26), 'market_value')
location = ctrl.Antecedent(np.arange(0, 11, 0.26), 'location')
house = ctrl.Consequent(np.arange(0, 11, 0.26), 'house')

market_value['low'] = fuzz.trapmf(market_value.universe, [0, 0, 80000, 100000])
market_value['medium'] = fuzz.trapmf(market_value.universe, [50000, 100000, 200000, 250000])
market_value['high'] = fuzz.trapmf(market_value.universe, [200000, 300000, 650000, 850000])
market_value['very_high'] = fuzz.trapmf(market_value.universe, [650000, 850000, 1000000, 1000000])

location['bad'] = fuzz.trapmf(location.universe, [0, 0, 2, 4])
location['fair'] = fuzz.trapmf(location.universe, [2.5, 5, 6, 8.5])
location['excellent'] = fuzz.trapmf(location.universe, [6, 8.5, 10, 10])

house['very_low'] = fuzz.trimf(house.universe, [0, 0, 3])
house['low'] = fuzz.trimf(house.universe, [0, 3, 6])
house['medium'] = fuzz.trimf(house.universe, [2, 5, 8])
house['high'] = fuzz.trimf(house.universe, [4, 7, 10])
house['very_high'] = fuzz.trimf(house.universe, [7, 10, 10])

rules_house = [
    ctrl.Rule(market_value['low'], house['low']),
    ctrl.Rule(location['bad'], house['low']),
    ctrl.Rule(location['bad'] & market_value['low'], house['very_low']),
    ctrl.Rule(location['bad'] & market_value['medium'], house['low']),
    ctrl.Rule(location['bad'] & market_value['high'], house['medium']),
    ctrl.Rule(location['bad'] & market_value['very_high'], house['high']),
    ctrl.Rule(location['fair'] & market_value['low'], house['low']),
    ctrl.Rule(location['fair'] & market_value['medium'], house['medium']),
    ctrl.Rule(location['fair'] & market_value['high'], house['high']),
    ctrl.Rule(location['fair'] & market_value['very_high'], house['very_high']),
    ctrl.Rule(location['excellent'] & market_value['low'], house['medium']),
    ctrl.Rule(location['excellent'] & market_value['medium'], house['high']),
    ctrl.Rule(location['excellent'] & market_value['high'], house['very_high']),
    ctrl.Rule(location['excellent'] & market_value['very_high'], house['very_high']),
]

house_ctrl = ctrl.ControlSystem(rules_house)
house_sim = ctrl.ControlSystemSimulation(house_ctrl)

# Applicant Evaluation
asset = ctrl.Antecedent(np.arange(0, 1000001, 0.26), 'asset')
income = ctrl.Antecedent(np.arange(0, 100001, 0.26), 'income')
applicant = ctrl.Consequent(np.arange(0, 11, 0.26), 'applicant')

asset['low'] = fuzz.trimf(asset.universe, [0, 0, 150000])
asset['medium'] = fuzz.trapmf(asset.universe, [50000, 250000, 450000, 650000])
asset['high'] = fuzz.trapmf(asset.universe, [500000, 700000, 1000000, 1000000])

income['low'] = fuzz.trapmf(income.universe, [0, 0, 12000, 25000])
income['medium'] = fuzz.trimf(income.universe, [15000, 35000, 55000])
income['high'] = fuzz.trimf(income.universe, [40000, 60000, 80000])
income['very_high'] = fuzz.trapmf(income.universe, [60000, 80000, 100000, 100000])

applicant['low'] = fuzz.trapmf(applicant.universe, [0, 0, 2, 4])
applicant['medium'] = fuzz.trimf(applicant.universe, [2, 5, 8])
applicant['high'] = fuzz.trapmf(applicant.universe, [6, 8, 10, 10])

rules_applicant = [
    ctrl.Rule(asset['low'] & income['low'], applicant['low']),
    ctrl.Rule(asset['low'] & income['medium'], applicant['low']),
    ctrl.Rule(asset['low'] & income['high'], applicant['medium']),
    ctrl.Rule(asset['low'] & income['very_high'], applicant['high']),
    ctrl.Rule(asset['medium'] & income['low'], applicant['low']),
    ctrl.Rule(asset['medium'] & income['medium'], applicant['medium']),
    ctrl.Rule(asset['medium'] & income['high'], applicant['high']),
    ctrl.Rule(asset['medium'] & income['very_high'], applicant['high']),
    ctrl.Rule(asset['high'] & income['low'], applicant['medium']),
    ctrl.Rule(asset['high'] & income['medium'], applicant['medium']),
    ctrl.Rule(asset['high'] & income['high'], applicant['high']),
    ctrl.Rule(asset['high'] & income['very_high'], applicant['high']),
]

applicant_ctrl = ctrl.ControlSystem(rules_applicant)
applicant_sim = ctrl.ControlSystemSimulation(applicant_ctrl)

# Credit Evaluation
interest = ctrl.Antecedent(np.arange(0, 11, 0.26), 'interest')
applicant_input = ctrl.Antecedent(np.arange(0, 11, 0.26), 'applicant_input')
house_input = ctrl.Antecedent(np.arange(0, 11, 0.26), 'house_input')
credit = ctrl.Consequent(np.arange(0, 500001, 0.26), 'credit')

interest['low'] = fuzz.trapmf(interest.universe, [0, 0, 2, 5])
interest['medium'] = fuzz.trapmf(interest.universe, [2, 4, 6, 8])
interest['high'] = fuzz.trapmf(interest.universe, [6, 8.5, 10, 10])

applicant_input['low'] = fuzz.trapmf(applicant_input.universe, [0, 0, 2, 4])
applicant_input['medium'] = fuzz.trimf(applicant_input.universe, [2, 5, 8])
applicant_input['high'] = fuzz.trapmf(applicant_input.universe, [6, 8, 10, 10])

house_input['very_low'] = fuzz.trimf(house_input.universe, [0, 0, 3])
house_input['low'] = fuzz.trimf(house_input.universe, [0, 3, 6])
house_input['medium'] = fuzz.trimf(house_input.universe, [2, 5, 8])
house_input['high'] = fuzz.trimf(house_input.universe, [4, 7, 10])
house_input['very_high'] = fuzz.trimf(house_input.universe, [7, 10, 10])

credit['very_low'] = fuzz.trimf(credit.universe, [0, 0, 125000])
credit['low'] = fuzz.trimf(credit.universe, [0, 125000, 250000])
credit['medium'] = fuzz.trimf(credit.universe, [125000, 250000, 375000])
credit['high'] = fuzz.trimf(credit.universe, [250000, 375000, 500000])
credit['very_high'] = fuzz.trimf(credit.universe, [375000, 500000, 500000])

credit.defuzzify_method = 'mom'

rules_credit = [
    ctrl.Rule(income['low'] & interest['medium'], credit['very_low']),
    ctrl.Rule(income['low'] & interest['high'], credit['very_low']),
    ctrl.Rule(income['medium'] & interest['high'], credit['low']),
    ctrl.Rule(applicant_input['low'], credit['very_low']),
    ctrl.Rule(house_input['very_low'], credit['very_low']),
    ctrl.Rule(applicant_input['medium'] & house_input['very_low'], credit['low']),
    ctrl.Rule(applicant_input['medium'] & house_input['low'], credit['low']),
    ctrl.Rule(applicant_input['medium'] & house_input['medium'], credit['medium']),
    ctrl.Rule(applicant_input['medium'] & house_input['high'], credit['high']),
    ctrl.Rule(applicant_input['medium'] & house_input['very_high'], credit['high']),
    ctrl.Rule(applicant_input['high'] & house_input['very_low'], credit['low']),
    ctrl.Rule(applicant_input['high'] & house_input['low'], credit['medium']),
    ctrl.Rule(applicant_input['high'] & house_input['medium'], credit['high']),
    ctrl.Rule(applicant_input['high'] & house_input['high'], credit['high']),
    ctrl.Rule(applicant_input['high'] & house_input['very_high'], credit['very_high'])
]

credit_ctrl = ctrl.ControlSystem(rules_credit)
credit_sim = ctrl.ControlSystemSimulation(credit_ctrl)

def calculate_credit_gui():
    try:
        mv = float(entry_mv.get())
        loc = float(entry_loc.get())
        asset_val = float(entry_asset.get())
        income_val = float(entry_income.get())
        interest_val = float(entry_interest.get())

        # House evaluation
        house_sim.input['market_value'] = mv
        house_sim.input['location'] = loc
        house_sim.compute()
        house_score = house_sim.output['house']

        mv_label = get_membership_label(mv, market_value)
        loc_label = get_membership_label(loc, location)
        house_label = get_membership_label(house_score, house)

        # Applicant evaluation
        applicant_sim.input['asset'] = asset_val
        applicant_sim.input['income'] = income_val
        applicant_sim.compute()
        applicant_score = applicant_sim.output['applicant']

        asset_label = get_membership_label(asset_val, asset)
        income_label = get_membership_label(income_val, income)
        applicant_label = get_membership_label(applicant_score, applicant)

        # Credit evaluation
        credit_sim.input['applicant_input'] = applicant_score
        credit_sim.input['house_input'] = house_score
        credit_sim.input['income'] = income_val
        credit_sim.input['interest'] = interest_val
        credit_sim.compute()
        credit_score = credit_sim.output['credit']

        interest_label = get_membership_label(interest_val, interest)
        credit_label = get_membership_label(credit_score, credit)

        credit_memberships = {
            label: fuzz.interp_membership(credit.universe, credit[label].mf, credit_score)
            for label in credit.terms
        }

        memberships_str = "\n".join([f"{key}: {value:.3f}" for key, value in credit_memberships.items()])

        result = (
            f"Market Value: {mv} → {mv_label}\n"
            f"Location: {loc} → {loc_label}\n"
            f"House Score: {house_score:.2f} → {house_label}\n\n"
            f"Asset: {asset_val} → {asset_label}\n"
            f"Income: {income_val} → {income_label}\n"
            f"Applicant Score: {applicant_score:.2f} → {applicant_label}\n\n"
            f"Interest Rate: {interest_val}% → {interest_label}\n\n"
            f"Credit Score: {credit_score:.2f} → {credit_label}\n"
            f"Credit membership degrees:\n{memberships_str}\n\n"
            f"Estimated Credit Amount: {credit_score:.2f}$"
        )

        messagebox.showinfo("Credit Result", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def get_membership_label(value, fuzzy_var):
    memberships = {
        label: fuzz.interp_membership(fuzzy_var.universe, fuzzy_var[label].mf, value)
        for label in fuzzy_var.terms
    }
    return max(memberships, key=memberships.get)

def view_memberships():
    try:
        market_value.view()
        location.view()
        house.view()
        asset.view()
        income.view()
        applicant.view()
        interest.view()
        applicant_input.view()
        house_input.view()
        credit.view()
    except Exception as e:
        messagebox.showerror("Error", f"Could not display plots:\n{e}")

# GUI with Tkinter
root = tk.Tk()
root.title("Credit Evaluation System")

tk.Label(root, text="Market Value (0 - 1,000,000$)").grid(row=0, column=0)
entry_mv = tk.Entry(root)
entry_mv.grid(row=0, column=1)

tk.Label(root, text="Location (0 - 10)").grid(row=1, column=0)
entry_loc = tk.Entry(root)
entry_loc.grid(row=1, column=1)

tk.Label(root, text="Asset (0 - 1,000,000$)").grid(row=2, column=0)
entry_asset = tk.Entry(root)
entry_asset.grid(row=2, column=1)

tk.Label(root, text="Income (0 - 100,000$)").grid(row=3, column=0)
entry_income = tk.Entry(root)
entry_income.grid(row=3, column=1)

tk.Label(root, text="Interest Rate (0 - 10%)").grid(row=4, column=0)
entry_interest = tk.Entry(root)
entry_interest.grid(row=4, column=1)

btn_calc = tk.Button(root, text="Calculate Credit", command=calculate_credit_gui)
btn_calc.grid(row=5, column=0, columnspan=2)

btn_view = tk.Button(root, text="View Membership Functions", command=view_memberships)
btn_view.grid(row=6, column=0, columnspan=2)

root.mainloop()
