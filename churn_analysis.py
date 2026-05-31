# ========== TASK 2: CUSTOMER RETENTION & CHURN ANALYSIS ==========
# pip install pandas matplotlib seaborn numpy plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ========== 1. CREATE CUSTOMER DATA ==========
np.random.seed(42)

n_customers = 2000
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

customer_data = []

for i in range(n_customers):
    signup_date = start_date + timedelta(days=np.random.randint(0, 180))
    subscription_type = np.random.choice(['Basic', 'Premium', 'Enterprise'], p=[0.5, 0.35, 0.15])
    monthly_fee = {'Basic': 29, 'Premium': 59, 'Enterprise': 99}[subscription_type]
    
    # Determine churn (30% overall churn rate)
    churned = np.random.random() < 0.3
    
    if churned:
        # Churn after 1-6 months
        months_active = np.random.randint(1, 7)
        churn_date = signup_date + timedelta(days=months_active * 30)
        if churn_date > end_date:
            churn_date = end_date
            churned = False
    else:
        months_active = np.random.randint(6, 12)
        churn_date = None
    
    # Customer engagement metrics
    avg_monthly_usage = np.random.randint(5, 100)
    support_tickets = np.random.poisson(0.5 if not churned else 2)
    payment_delay = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) if churned else 0
    
    customer_data.append({
        'customer_id': f'CUST_{i+1:04d}',
        'signup_date': signup_date,
        'churned': churned,
        'churn_date': churn_date,
        'subscription_type': subscription_type,
        'monthly_fee': monthly_fee,
        'months_active': months_active,
        'avg_monthly_usage_hours': avg_monthly_usage,
        'support_tickets': support_tickets,
        'payment_delay_days': payment_delay,
        'referral_source': np.random.choice(['Google', 'Friend', 'LinkedIn', 'Email', 'Other'], p=[0.3, 0.25, 0.2, 0.15, 0.1])
    })

df_customers = pd.DataFrame(customer_data)

print("=" * 60)
print("TASK 2: CUSTOMER RETENTION & CHURN ANALYSIS")
print("=" * 60)
print(f"\nTotal Customers: {len(df_customers)}")
print(f"Churned Customers: {df_customers['churned'].sum()}")
print(f"Churn Rate: {df_customers['churned'].mean()*100:.1f}%")
print(f"Active Customers: {(~df_customers['churned']).sum()}")

# ========== 2. COHORT ANALYSIS ==========
# Create cohort groups by signup month
df_customers['signup_month'] = df_customers['signup_date'].dt.to_period('M')

# Simulate monthly retention
cohort_data = []
for cohort in df_customers['signup_month'].unique():
    cohort_customers = df_customers[df_customers['signup_month'] == cohort]
    total_cohort = len(cohort_customers)
    
    retention_rates = []
    for month in range(1, 7):  # Track first 6 months
        retained = sum((c['months_active'] >= month) for _, c in cohort_customers.iterrows())
        retention_rates.append(retained / total_cohort * 100)
    
    cohort_data.append({
        'Cohort': str(cohort),
        'Month 1': retention_rates[0],
        'Month 2': retention_rates[1] if len(retention_rates) > 1 else 0,
        'Month 3': retention_rates[2] if len(retention_rates) > 2 else 0,
        'Month 4': retention_rates[3] if len(retention_rates) > 3 else 0,
        'Month 5': retention_rates[4] if len(retention_rates) > 4 else 0,
        'Month 6': retention_rates[5] if len(retention_rates) > 5 else 0,
    })

df_cohort = pd.DataFrame(cohort_data)

# Plot cohort retention heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CUSTOMER RETENTION & CHURN ANALYSIS DASHBOARD', fontsize=15, fontweight='bold')
# Heatmap
cohort_matrix = df_cohort.set_index('Cohort').values
im = axes[0, 0].imshow(cohort_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
axes[0, 0].set_xticks(range(len(df_cohort.columns[1:])))
axes[0, 0].set_xticklabels(df_cohort.columns[1:], rotation=45)
axes[0, 0].set_yticks(range(len(df_cohort)))
axes[0, 0].set_yticklabels(df_cohort['Cohort'])
axes[0, 0].set_title('Cohort Retention Heatmap (%)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Months Since Signup')
axes[0, 0].set_ylabel('Signup Cohort')
plt.colorbar(im, ax=axes[0, 0])

# Churn by subscription type
churn_by_type = df_customers.groupby('subscription_type')['churned'].mean() * 100
bars = axes[0, 1].bar(churn_by_type.index, churn_by_type.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_title('Churn Rate by Subscription Type', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_by_type.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', fontweight='bold')

# Churn drivers - Support tickets correlation
churned_cust = df_customers[df_customers['churned'] == True]
retained_cust = df_customers[df_customers['churned'] == False]

ticket_data = [retained_cust['support_tickets'], churned_cust['support_tickets']]
bp = axes[1, 0].boxplot(ticket_data, labels=['Retained', 'Churned'], patch_artist=True)
bp['boxes'][0].set_facecolor('#90EE90')
bp['boxes'][1].set_facecolor('#FFB6C1')
axes[1, 0].set_title('Support Tickets: Retained vs Churned', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Number of Support Tickets')

# Payment delay distribution
payment_delay_churn = df_customers[df_customers['churned']]['payment_delay_days']
payment_delay_retained = df_customers[~df_customers['churned']]['payment_delay_days']
axes[1, 1].hist(payment_delay_retained, alpha=0.7, bins=3, label='Retained', color='green')
axes[1, 1].hist(payment_delay_churn, alpha=0.7, bins=3, label='Churned', color='red')
axes[1, 1].set_title('Payment Delay Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Payment Delay (days)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('churn_analysis_1.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3. CUSTOMER LIFETIME VALUE (CLV) ==========
df_customers['clv'] = df_customers['monthly_fee'] * df_customers['months_active']

print("\n" + "=" * 60)
print("CUSTOMER LIFETIME VALUE ANALYSIS")
print("=" * 60)
print(f"\nAverage CLV: ${df_customers['clv'].mean():.2f}")
print(f"Median CLV: ${df_customers['clv'].median():.2f}")
print(f"Retained Customers Avg CLV: ${df_customers[~df_customers['churned']]['clv'].mean():.2f}")
print(f"Churned Customers Avg CLV: ${df_customers[df_customers['churned']]['clv'].mean():.2f}")

# CLV by subscription type
print("\nCLV by Subscription Type:")
for sub_type in df_customers['subscription_type'].unique():
    avg_clv = df_customers[df_customers['subscription_type'] == sub_type]['clv'].mean()
    print(f"  • {sub_type}: ${avg_clv:.2f}")

# ========== 4. KEY INSIGHTS ==========
print("\n" + "=" * 60)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("\n📉 CHURN PATTERNS:")
print(f"  • Highest churn: Basic plan ({churn_by_type['Basic']:.1f}%)")
print(f"  • Lowest churn: Enterprise ({churn_by_type['Enterprise']:.1f}%)")
print(f"  • Average customer lifespan: {df_customers['months_active'].mean():.1f} months")
print(f"  • Churned customers have {churned_cust['support_tickets'].mean():.1f} tickets vs {retained_cust['support_tickets'].mean():.1f} for retained")

print("\n💡 ACTIONABLE RECOMMENDATIONS:")
print("  ✅ Implement early warning system for customers with 2+ support tickets")
print("  ✅ Reduce payment friction - offer auto-pay discount for Basic tier")
print("  ✅ Launch retention campaign at month 3 (critical churn period)")
print("  ✅ Upsell Basic users to Premium with free trial + feature showcase")