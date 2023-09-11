## 🎒📘 CommonLit - Evaluate Student Summaries

### 🖍️ Final Score

- Private Score:
- Rank:

### 🗂️ Competition Summary

**To create an automated assessment model for students' summative assignments**


### 💡 Strategy
`🎨 Various Embedding Ensemble`

![Modeling Strategy](/assets/modeling_strategy.png)

- **🖍️ Make Two types of Prompt for LLM's Input**

    **`Type 1) OneToOne Schema Prompt`**
    - **~~1-1) summaries_text + prompt_question + prompt_title + prompt_text~~**
        - **Distribution of Train/Validation/Test does not match when useing this prompt, caused mismatch between CV Score and Competition Score**
    - **1-2) summaries_text + prompt_question + prompt_title**
    - **~~1-3) summaries_text + prompt_question + prompt_text~~**
    - **~~1-4) summaries_text + prompt_title + prompt_text~~**
    
    **`Type 2) OneToMany Schema Prompt`**
    - **`OneToMany Prompt`: [ANC] p_question [ANC] p_title [ANC] [SEP] [TAR] s_text 1 [TAR] s_text 2 [TAR] s_text 3 [TAR] ... [TAR] s_text N [TAR] [SEP]**
        - **[ANC]: Special Token, which is meaning of 'anchor', x**
        - **[TAR]: Special Token, which is meaning of 'target', y**
    - **`Masking Tensor`: tensor object for masking unnecessary element in output embeddings, only remaining target text(s_text)**
        - **[0, 0, 0, 0, 0, ... 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ... 0, 1, 1, 1, 1, 0, 0]**
    
    **`Common Apply`**
    - **`AutoSpellChecker`: for correcting mis-spell words in original text**
        - **mis-spell doesn't affect target scores which is officially announced by the host**
    - **`Smart Batching System`: for decreasing train time & memory usage**
    - **`No Padding & Truncating`: for preserving original context embedding**

### 🔗 References
##### 1) P-Tuning: https://arxiv.org/abs/2103.10385
##### 2) Longformer: https://arxiv.org/abs/2004.05150
##### 3) DeBERTa: https://arxiv.org/abs/2006.03654
