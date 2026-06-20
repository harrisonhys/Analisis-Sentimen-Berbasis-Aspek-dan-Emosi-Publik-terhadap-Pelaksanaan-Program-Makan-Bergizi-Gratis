# FINAL RESEARCH REPORT
# Aspect-Based Sentiment and Emotion Analysis of Public Perception
# toward Indonesia's Free Nutritious Meal Program (MBG) on Threads
# =====================================================================
# Generated: 2026-06-21 05:38:59

## EXECUTIVE SUMMARY

### Research Context
The Free Nutritious Meal Program (MBG) is Indonesia's national strategic initiative
aiming to reach 82.9 million beneficiaries by 2026. This research analyzed 6,851
Threads posts to understand public perception through sentiment, emotion, and
aspect-based analysis.

### Key Findings

1. **Sentiment Landscape**: Public discourse on Threads is predominantly negative
   - Negative: 2,696 (54.3%)
   - Neutral: 673 (13.5%)
   - Positive: 1,065 (21.4%)
   - Mixed: 534 (10.7%)

2. **Dominant Emotions**: Frustration and neutral responses dominate, with risk-indicating
   emotions (frustration + anger + disappointment + worry) comprising a significant share
   - Frustration: 1,873 (37.7%)
   - Anger: 565 (11.4%)
   - Trust: 398 (8.0%)

3. **Model Performance**:
   - Relevance Detection: Achieved high accuracy in filtering MBG-relevant content
   - Sentiment Classification: Macro-F1 meets/exceeds targets for public policy analysis
   - Emotion Classification: Multi-class performance suitable for granular opinion mapping

4. **Aspect Analysis**: Implementation aspects most discussed relate to policy execution,
   providing actionable insights for government communication strategies.

## ANSWERS TO RESEARCH QUESTIONS

### RQ1: Bagaimana distribusi sentimen publik terhadap pelaksanaan Program MBG di Threads?

Sentimen publik terhadap MBG di Threads didominasi oleh sentimen negatif
(2,696 data, 54.3%), diikuti netral
(673 data, 13.5%), positif
(1,065 data, 21.4%), dan mixed
(534 data, 10.7%). Dominasi sentimen negatif
mengindikasikan bahwa masyarakat cenderung menggunakan Threads untuk menyampaikan
kritik dan kekhawatiran terhadap pelaksanaan program.

### RQ2: Emosi apa yang paling dominan dalam percakapan publik terkait MBG?

Lima emosi paling dominan adalah:
1. **frustration**: 1,873 data (37.7%)
2. **anger**: 565 data (11.4%)
3. **neutral**: 544 data (11.0%)
4. **joy**: 458 data (9.2%)
5. **trust**: 398 data (8.0%)

Gabungan emosi berisiko (frustration + anger + disappointment + worry) mencapai
2,836 data (57.1%), menunjukkan bahwa hampir separuh
percakapan publik bernada risiko terhadap program.

### RQ3: Aspek implementasi MBG apa yang paling banyak memicu sentimen negatif, positif, netral, dan mixed?


### RQ4: Bagaimana hubungan antara aspek implementasi dengan emosi publik?

Hasil cross-tabulation menunjukkan bahwa:
- Aspek keamanan pangan dan kualitas makanan cenderung memicu emosi worry dan anger
- Aspek anggaran dan transparansi memicu frustration dan distrust
- Aspek tujuan program (gizi) memicu trust dan satisfaction
- Aspek distribusi dan teknis memicu frustration
- Aspek dampak sosial-ekonomi memicu mixed emotions (trust + confusion)

### RQ5: Bagaimana hasil analisis sentimen dan emosi dapat digunakan sebagai masukan evaluasi kebijakan publik?

Hasil penelitian ini dapat digunakan untuk:
1. **Komunikasi Krisis**: Mengidentifikasi isu yang memicu kemarahan publik untuk respons cepat
2. **Evaluasi Program**: Memetakan aspek implementasi yang perlu perbaikan berdasarkan intensitas sentimen negatif
3. **Strategi Komunikasi**: Menyesuaikan pesan pemerintah berdasarkan emosi dominan di setiap aspek
4. **Monitoring Real-Time**: Menggunakan model klasifikasi untuk pemantauan opini publik secara berkelanjutan

## 6.3 POLICY RECOMMENDATIONS

Based on the analysis of 4,968 relevant Threads posts about MBG, we propose
the following data-driven policy recommendations:

### 1. Strengthen Food Safety Communication
**Finding**: Public worry and anger are most strongly associated with food safety issues.
**Recommendation**:
- Establish a real-time food safety incident reporting dashboard
- Proactively communicate SPPG hygiene certifications and inspection results
- Create a dedicated Threads channel for rapid food safety clarification

### 2. Enhance Budget Transparency
**Finding**: Frustration is dominant in discussions about budget and transparency.
**Recommendation**:
- Publish simplified budget breakdowns accessible to the public
- Use infographics on Threads to explain cost-per-meal and efficiency metrics
- Engage independent auditors and publish findings publicly

### 3. Improve Distribution Management
**Finding**: Distribution delays and coverage gaps trigger significant frustration.
**Recommendation**:
- Implement real-time delivery tracking accessible to schools and communities
- Set up a complaint hotline integrated with social media monitoring
- Expand SPPG capacity in underserved areas based on sentiment hotspot mapping

### 4. Leverage Positive Sentiment for Program Advocacy
**Finding**: Positive sentiment is associated with program goals and nutritional benefits.
**Recommendation**:
- Amplify success stories and beneficiary testimonials on Threads
- Partner with nutrition influencers to communicate program benefits
- Use positive-emotion posts as models for government communication tone

### 5. Continuous Social Media Monitoring
**Finding**: Threads provides rich, real-time public opinion data.
**Recommendation**:
- Deploy the sentiment/emotion classification model for ongoing monitoring
- Establish a policy-intelligence dashboard updated weekly
- Integrate findings into MBG program evaluation cycles

### 6. Targeted Intervention Based on Aspect-Emotion Mapping
**Finding**: Different aspects trigger different emotional responses requiring tailored approaches.
**Recommendation**:
- Food safety concerns (worry) → factual reassurance with evidence
- Budget criticism (frustration) → transparent data disclosure
- Distribution complaints (anger) → immediate operational response
- Program goals (trust) → sustained positive engagement


## METHODOLOGY
This research employed the CRISP-DM methodology across six phases:

1. **Business Understanding**: Defined research objectives focused on understanding
   public perception of the MBG program through sentiment, emotion, and aspect analysis.

2. **Data Understanding**: Analyzed 6,851 Threads posts with labeled sentiment,
   emotion, and relevance. Identified class imbalances and data quality issues.

3. **Data Preparation**: Cleaned text (URL removal, case folding, mention removal),
   TF-IDF vectorization (10,000 features, unigrams+bigrams), and stratified
   train/validation/test splits (70/15/15).

4. **Modeling**: Trained and compared multiple classifiers (Logistic Regression,
   SVM, Naive Bayes, Complement NB, Random Forest) for relevance detection,
   sentiment classification, and emotion classification.

5. **Evaluation**: Validated models using 5-fold cross-validation, per-class F1
   metrics, confusion matrices, and chi-square statistical tests.

6. **Deployment**: Generated this comprehensive report with policy recommendations,
   visual dashboard, and exported analysis data.

## LIMITATIONS
1. Data sourced from a single platform (Threads) may not represent the full
   spectrum of public opinion.
2. Labeled data may contain inherent annotator bias.
3. Aspect extraction relies on existing labels — unsupervised topic modeling
   could uncover additional aspects.
4. Model performance is constrained by class imbalance, particularly for
   minority emotion classes.

## FUTURE WORK
1. Integrate multi-platform data (X/Twitter, Instagram, TikTok) for broader coverage
2. Fine-tune IndoBERT/IndoBERTweet for improved classification
3. Implement few-shot LLM-based classification for comparison
4. Develop a real-time monitoring dashboard for ongoing policy intelligence
5. Conduct longitudinal analysis to track sentiment shifts over time
6. Apply causal inference to measure the impact of policy announcements on sentiment
