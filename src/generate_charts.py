import matplotlib.pyplot as plt
import os

os.makedirs('outputs/figures', exist_ok=True)

experiments = ['Exp 1\n(FT r=16)', 'Exp 2\n(3 epochs)', 'Exp 4\n(FT r=32)', 'Exp 5\n(lr=3e-4)',
               'Exp 7\n(prefix)', 'Exp 10\n(retrieval)', 'Exp 11\n(lang-aware)', 'Exp 12\n(top-3)',
               'Exp 13\n(mpnet)', 'Exp 14\n(q+a embed)', 'Exp 15\n(RAG)', 'Exp 16\n(BM25)']
scores = [0.182709, 0.095423, 0.194928, 0.132925, 0.195404, 0.479688, 0.486525,
          0.411156, 0.510140, 0.432645, 0.228704, 0.461861]

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['steelblue'] * 5 + ['coral'] * 7
ax.bar(experiments, scores, color=colors)
ax.axhline(y=0.510140, color='green', linestyle='--', alpha=0.5, label='Best score (0.510140)')
ax.set_ylabel('Zindi Score')
ax.set_title('Experiment Score Progression: Fine-Tuning vs Retrieval Approaches')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/experiment_score_progression.png', dpi=150)
print('Saved experiment_score_progression.png')

languages = ['Aka_Gha', 'Eng_Gha', 'Eng_Eth', 'Swa_Ken', 'Eng_Ken', 'Eng_Uga', 'Lug_Uga', 'Amh_Eth']
rouge1 = [0.336, 0.295, 0.225, 0.191, 0.173, 0.162, 0.128, 0.011]
rougeL = [0.234, 0.232, 0.185, 0.149, 0.127, 0.117, 0.106, 0.011]

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(languages))
ax.bar([i - 0.2 for i in x], rouge1, width=0.4, label='ROUGE-1', color='steelblue')
ax.bar([i + 0.2 for i in x], rougeL, width=0.4, label='ROUGE-L', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(languages, rotation=45, ha='right')
ax.set_ylabel('ROUGE Score')
ax.set_title('Per-Language Performance (Experiment 8, Experiment 7 model on validation set)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/per_language_rouge.png', dpi=150)
print('Saved per_language_rouge.png')
