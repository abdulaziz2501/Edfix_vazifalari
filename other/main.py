import seaborn as sns
import matplotlib.pyplot as plt

# Masalan, ichki ma'lumotlar to'plamidan foydalanamiz
tips = sns.load_dataset("tips")

# Seaborn bilan grafik chizamiz
sns.barplot(x="day", y="total_bill", data=tips)
plt.title("Kunlar bo'yicha hisoblar")
plt.show()