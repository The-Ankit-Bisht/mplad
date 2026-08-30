import os
import re
import pandas as pd
import numpy as np

def extract_work_id(df, col_name='work'):
    if col_name in df.columns:
        df['work_clean'] = df[col_name].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df['work_id'] = df['work_clean'].str.extract(r'^(WS/\s*MP\d+/\d{4}-\d{4}/\d+)')[0].str.replace(r"\s+", "", regex=True)
        df.drop(columns=['work_clean'], inplace=True)
    return df

def run_data_pipeline():
    print("Starting data processing pipeline...")
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    recommended = pd.read_csv('data/raw/works_recommended.csv')
    sanctioned = pd.read_csv('data/raw/works_sanctioned.csv')
    completed = pd.read_csv('data/raw/works_completed.csv')
    expenditure = pd.read_csv('data/raw/expenditure.csv')
    
    # Extract Work IDs
    recommended = extract_work_id(recommended)
    sanctioned = extract_work_id(sanctioned)
    completed = extract_work_id(completed)
    expenditure = extract_work_id(expenditure, 'work_id')
    
    # Aggregate Expenditure
    expenditure['Expenditure Date'] = pd.to_datetime(expenditure['Expenditure Date'], format='%d-%b-%Y', errors='coerce')
    exp_agg = expenditure.groupby('work_id').agg(
        total_disbursed=('Fund Disbursed Amount ( ₹ )', 'sum'),
        payment_count=('work_id', 'size'),
        successful_payments=('Payment Status', lambda x: (x == 'Payment Success').sum()),
        failed_pending_payments=('Payment Status', lambda x: (x != 'Payment Success').sum()),
        first_payment_date=('Expenditure Date', 'min'),
        latest_payment_date=('Expenditure Date', 'max')
    ).reset_index()
    
    # Subset valid Work IDs
    sanc_ws = sanctioned[sanctioned['work_id'].notna()].copy()
    rec_ws = recommended[recommended['work_id'].notna()].copy()
    comp_ws = completed[completed['work_id'].notna()].copy()
    
    # Master Merge
    master_df = sanc_ws[['work_id', 'State', 'Constituency', "Hon'ble Members of Parliament", 'Sanction Amount ( ₹ )', 'Sanction Date', 'Work Status']].copy()
    master_df = master_df.merge(
        rec_ws[['work_id', 'Work category', 'RECOMMENDED AMOUNT ( ₹ )', 'Recommended date', 'Work description']], 
        on='work_id', how='left'
    )
    master_df = master_df.merge(
        comp_ws[['work_id', 'Amount Disbursed ( ₹ )', 'Completion Date']], 
        on='work_id', how='left'
    )
    master_df = master_df.merge(exp_agg, on='work_id', how='left')
    
    # Date parsing and calculations
    for col in ['Sanction Date', 'Recommended date', 'Completion Date']:
        master_df[col] = pd.to_datetime(master_df[col], format="%d-%b-%Y", errors="coerce")
        
    master_df['is_completed'] = master_df['Completion Date'].notna().astype(int)
    master_df['total_disbursed'] = master_df['total_disbursed'].fillna(0)
    master_df['days_rec_to_sanc'] = (master_df['Sanction Date'] - master_df['Recommended date']).dt.days
    master_df['days_active'] = (pd.Timestamp.now() - master_df['Sanction Date']).dt.days
    
    # Target Ratios & Target Flags
    master_df['fund_utilization_ratio'] = np.where(
        master_df['Sanction Amount ( ₹ )'] > 0, 
        master_df['total_disbursed'] / master_df['Sanction Amount ( ₹ )'], 0
    )
    master_df['is_delayed'] = np.where((master_df['is_completed'] == 0) & (master_df['days_active'] > 365), 1, 0)
    master_df['is_high_risk'] = np.where((master_df['is_delayed'] == 1) & (master_df['fund_utilization_ratio'] < 0.2), 1, 0)
    
    master_df = master_df.fillna({
        'Sanction Amount ( ₹ )': 0, 'RECOMMENDED AMOUNT ( ₹ )': 0, 
        'payment_count': 0, 'successful_payments': 0, 'failed_pending_payments': 0
    })
    
    # Advanced Historical Aggregations
    mp_stats = master_df.groupby("Hon'ble Members of Parliament")['fund_utilization_ratio'].mean().reset_index()
    mp_stats.rename(columns={'fund_utilization_ratio': 'mp_avg_utilization'}, inplace=True)
    master_df = master_df.merge(mp_stats, on="Hon'ble Members of Parliament", how='left')
    
    state_stats = master_df.groupby('State')['is_delayed'].mean().reset_index()
    state_stats.rename(columns={'is_delayed': 'state_delay_rate'}, inplace=True)
    master_df = master_df.merge(state_stats, on='State', how='left')
    
    master_df['amount_diff'] = master_df['Sanction Amount ( ₹ )'] - master_df['RECOMMENDED AMOUNT ( ₹ )']
    
    # Export
    output_path = 'data/processed/master_df.csv'
    master_df.to_csv(output_path, index=False)
    print(f"Data pipeline complete! Processed file saved to {output_path}. Shape: {master_df.shape}")

if __name__ == "__main__":
    run_data_pipeline()