# **BT4222 Final Report**

**SPOTLIGHT: A News Recommendation Framework for Homepage Slate Optimisation**

**Group 12: Chin Li Wen, Jing Xinai, Ng Jia Siang Jason, Nguyen Nhat Tan,**   
**Nguyen Van Minh, Phua Xing Xi Charlene.**

# **Abstract (Standalone page, within 300 words)**

* **Business Objective & RecSys Formulation:** Define the core recommendation problem grounded in an exploratory understanding of dataset distributions and domain constraints.  
  * Top K & cold start  
* **Prior Baseline & Critical Diagnosis:** Frame your starting point with reference to past project baselines (e.g., NCF, NMF), identifying their structural limitations, methodological gaps, or performance bottlenecks.  
* **Group Contributions & Modeling Highlights:** Summarize your group's engineering, architectural, and analytical improvements over the baseline.  
* **Evidence-Backed Quantitative Insights:** State key business and behavioral insights supported directly by concrete quantitative metrics, offline evaluation, or diagnostic error analysis

# **1\. Data Understanding & Feature Engineering**

Instead of repeating the interim report, focus on updating changes, data distributions, and incremental features developed (e.g., utilizing impression feedback, interaction types, or context signals). Use structured tables where appropriate.

- Changes: Remove slate\_length \+ article\_recency (Give explanation)  
- Incremental features: Missingness Indicator   
- Modify the tables 

# **2\. Models and Performance (Within 4 pages)**

## **2.1 Architecture Diagrams**

Provide clear, modular diagrams illustrating model architecture, input pipelines, and data flow

## **2.2 Performance Reporting**

Report both training and testing performance across baseline and advanced models using appropriate ranking metrics with clear explanations of sample splits.

# **3\. Contribution and Justification (1–3 pages per chosen aspect)**

Complete the Self-Evaluation Table below. Groups are not required to cover all optional aspects; focus on the areas matching your team's technical depth.

| Contribution Aspect  | Dimension | Basic | Optional / Advanced |
| :---- | :---- | :---- | :---- |
| **1\. Data Understanding, Rich Signal Utilization & Feature Engineering** (Impressions, multi-type user interactions, context, domain features) | **Effort** |  |  |
|  | **Effectiveness** |  |   |
| *Clarification:* **Basic:** e.g. Convert data into standard binary interaction pairs without deeper exploration of additional information; apply standard transformations or descriptive statistics. **Optional:** e.g. Thoroughly analyze data distributions and sparsity prior to modeling; exploit rich signals (e.g., impression logs, exposed vs. unexposed items, multi-behavior interactions, timestamps/sessions) and engineer domain-informed features to enrich predictive signals. |  |  |  |
| **2\. Prior Baseline Benchmarking & Gap Diagnosis**  (Critical evaluation of NCF, NMF, and conventional baselines) | **Effort** |  |  |
|  | **Effectiveness** |  |   |
| *Clarification:* **Basic:** e.g. Replicate standard baseline models (e.g., NCF, NMF) directly to obtain comparative metrics. **Optional:** e.g. Critically diagnose structural failure modes or information bottlenecks in baseline implementations; systematically justify your improvements against these identified gaps.	 |  |  |  |
| **3\. Designing or Adapting Recommender Architectures** | **Effort** |  |  |
|  | **Effectiveness** |  |   |
| *Clarification:* **Basic:** e.g. Direct implementation of standard models or simple ensembling. **Optional:** e.g. Adapt or customize architectures capturing temporal dynamics, sequential interaction patterns, or network/relational structures to improve recommendation quality.	 |  |  |  |
| **4\. Model Diagnostics, Error Analysis & Quantitative Insights** | **Effort** |  |  |
|  | **Effectiveness** |  |   |
| *Clarification:* **Basic:** e.g. Describe aggregate test metrics without deeper sub-group analysis **Optional:** e.g. Perform granular error diagnostics across distinct user/item segments; conduct deep learning model interpretation by examining learned parameters (e.g., analyzing attention weights, etc.); extract actionable behavioral and business insights supported by empirical evidence	 |  |  |  |
| **5\. Smart Use of AI Tools & Critical Collaboration** | **Effort** |  |  |
|  | **Effectiveness** |  |   |
| *Clarification:* **Basic:** Use AI tools strictly for routine coding syntax, formatting, or text polishing **Optional:** Engage AI as a critical sparring partner to explore modeling alternatives, catch conceptual/coding flaws, validate technical logic, and drive expanded learning documented in the AI Interaction Log.	 |  |  |  |

# **4\. AI Interaction Log (Appendix)**

Provide a concise table documenting key collaborative interactions with AI tools:  
**1\. Core Prompt & Task Objective:** The specific question, modeling concept, or technical problem being explored.  
**2\. AI Output Summary:** The substantive response or code generated (do not include all of the code generated by AI in the Appendix).  
**3\. Critical Evaluation:** Your assessment of limitations, naive assumptions, or errors in the AI  
response.  
**4\. Refinement & Final Action:** How you validated, adjusted, and incorporated the solution into your project

# **5\. References (Within 1 page)**

Comprehensive citations for all datasets, baseline repositories, and referenced academic literature.