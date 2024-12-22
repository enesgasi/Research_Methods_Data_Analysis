#libraries used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway


review_scores_file = "datasets/sales_with_review_scores.csv"
vgsales_file = "datasets/vgsales.csv"

review_data = pd.read_csv(review_scores_file)
vgsales_data = pd.read_csv(vgsales_file)

# --- H1: Genre popularity differs significantly across regions ---

action_sales_na = vgsales_data[vgsales_data['Genre'] == 'Action']['NA_Sales']
action_sales_eu = vgsales_data[vgsales_data['Genre'] == 'Action']['EU_Sales']
action_sales_jp = vgsales_data[vgsales_data['Genre'] == 'Action']['JP_Sales']
action_sales_other = vgsales_data[vgsales_data['Genre'] == 'Action']['Other_Sales']

f_stat, p_value = f_oneway(action_sales_na, action_sales_eu, action_sales_jp, action_sales_other)


print("H1: Genre popularity differs significantly across regions")
print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.2e}")
print("Conclusion: Genre popularity differs significantly across regions at the 0.05 significance level.\n")


region_sales_by_genre = vgsales_data.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()


plt.figure(figsize=(12, 8))
region_sales_by_genre.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')
plt.title('Regional Sales by Genre (H1: Regional Differences)')
plt.ylabel('Total Sales (Millions)')
plt.xlabel('Genre')
plt.xticks(rotation=45)
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(region_sales_by_genre, annot=True, fmt=".1f", cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Regional Sales by Genre (H1: Regional Differences)')
plt.ylabel('Genre')
plt.xlabel('Region')
plt.tight_layout()
plt.show()

# --- H2: Higher review scores lead to significantly better sales ---

median_critic_score = review_data['Critic_Score'].median()
high_score_sales = review_data[review_data['Critic_Score'] >= median_critic_score]['Global_Sales']
low_score_sales = review_data[review_data['Critic_Score'] < median_critic_score]['Global_Sales']

t_stat_critic, p_value_critic = ttest_ind(high_score_sales, low_score_sales, nan_policy='omit')


print("H2: Higher review scores lead to significantly better sales")
print(f"T-statistic: {t_stat_critic:.2f}")
print(f"p-value: {p_value_critic:.2e}")
print("Conclusion: Games with higher critic scores have significantly better sales at the 0.05 significance level.\n")


plt.figure(figsize=(10, 6))
sns.scatterplot(data=review_data, x='Critic_Score', y='Global_Sales', alpha=0.7, color='blue', edgecolor='black')
plt.title('Global Sales vs. Critic Score (H2)')
plt.xlabel('Critic Score')
plt.ylabel('Global Sales (Millions)')
plt.tight_layout()
plt.show()

# --- H3: Action and adventure games have significantly higher global sales ---

action_sales = vgsales_data[vgsales_data['Genre'] == 'Action']['Global_Sales']
adventure_sales = vgsales_data[vgsales_data['Genre'] == 'Adventure']['Global_Sales']
other_genres_sales = vgsales_data[~vgsales_data['Genre'].isin(['Action', 'Adventure'])]['Global_Sales']

t_stat_action, p_value_action = ttest_ind(action_sales, other_genres_sales, nan_policy='omit')
t_stat_adventure, p_value_adventure = ttest_ind(adventure_sales, other_genres_sales, nan_policy='omit')


print("H3: Action and adventure games have significantly higher global sales than other genres")
print("Action vs Other Genres:")
print(f"T-statistic: {t_stat_action:.2f}")
print(f"p-value: {p_value_action:.2e}")
print("Conclusion: Action games do not have significantly higher global sales compared to other genres at the 0.05 significance level.\n")

print("Adventure vs Other Genres:")
print(f"T-statistic: {t_stat_adventure:.2f}")
print(f"p-value: {p_value_adventure:.2e}")
print("Conclusion: Adventure games have significantly lower global sales compared to other genres at the 0.05 significance level.\n")


total_sales = pd.DataFrame({
    'Genre': ['Action', 'Adventure', 'Other Genres'],
    'Total_Global_Sales': [
        action_sales.sum(),
        adventure_sales.sum(),
        other_genres_sales.sum()
    ]
})


plt.figure(figsize=(10, 6))
sns.barplot(data=total_sales, x='Genre', y='Total_Global_Sales', hue='Genre', palette='Set2', dodge=False, edgecolor='black')
plt.title('Total Global Sales: Action, Adventure, and Other Genres (H3)')
plt.ylabel('Total Global Sales (Millions)')
plt.xlabel('Genre Category')
plt.legend([], [], frameon=False)  # Hide legend since it's redundant
plt.tight_layout()
plt.show()


vgsales_data['Genre_Category'] = vgsales_data['Genre'].apply(
    lambda x: 'Action' if x == 'Action' else 'Adventure' if x == 'Adventure' else 'Other Genres'
)

plt.figure(figsize=(10, 6))
sns.boxplot(data=vgsales_data, x='Genre_Category', y='Global_Sales', hue='Genre_Category', palette='Set3', dodge=False)
plt.title('Distribution of Global Sales: Action, Adventure, and Other Genres (H3)')
plt.ylabel('Global Sales (Millions)')
plt.xlabel('Genre Category')
plt.legend([], [], frameon=False)  # Hide legend since it's redundant
plt.tight_layout()
plt.show()
