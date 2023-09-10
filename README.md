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

    **Type 1) `OneToOne` Schema Prompt**
    - **~~1-1) summaries_text + prompt_question + prompt_title + prompt_text~~**
        - **Distribution of Train/Validation/Test does not match when useing this prompt, caused mismatch between CV Score and Competition Score**
    - **1-2) summaries_text + prompt_question + prompt_title**
    - **~~1-3) summaries_text + prompt_question + prompt_text~~**
    - **~~1-4) summaries_text + prompt_title + prompt_text~~**
    
    **Type 2) `OneToMany` Schema Prompt**
    - **2-1) prompt_question + prompt_title + summaries_text1 + summaries_text2 + ... + summaries_textN**

    **Common Apply**
    - `AutoSpellChecker` for correcting mis-spell words in original text
        - mis-spell doesn't affect target scores which is officially announced by the host
    - `Smart Batching System` for decreasing train time & memory usage
    - `No Padding & Truncating` for preserving original context embedding

### 🔗 References
