
# Data Cleansing Scripts

This folder contains scripts for cleaning the PartIII agent data and ranking pairs.

## Files

### `clean_agent_frequency.py`

This script reads the raw PartIII agent file:

```bash
../dataset/PartIII/agents/merge.json
````

It cleans noisy LLM and tool names, then outputs cleaned agents and frequency statistics into this folder.

Main outputs:

```bash
merge.cleaned.json
llm_frequency.json
tool_frequency.json
llm_count.json
tool_count.json
frequency_cleansing_summary.json
```

Cleaning rules:

* LLM names containing `part` are treated as noisy and replaced with `Default_LLM`.
* Tool names are cleaned more conservatively. Only obvious `PartI`, `PartII`, or `PartIII` style tool names are replaced with `Default_Tool`.

Run:

```bash
python clean_agent_frequency.py
```

---

### `create_partiii_cleaned.py`

This script reads:

```bash
merge.cleaned.json
../dataset/PartIII/rankings/merge.json
```

It removes invalid question-agent pairs from the rankings. An agent is treated as invalid if it uses `Default_LLM`, uses `Default_Tool`, or is missing from the cleaned agent file.

Main outputs:

```bash
PartIII_cleaned/agents/merge.json
PartIII_cleaned/rankings/merge.json
PartIII_cleaned/cleaning_summary.json
```

Run:

```bash
python create_partiii_cleaned.py
```

## Output Folder

After running both scripts, this folder will contain cleaned agents, frequency statistics, and a cleaned PartIII dataset with invalid ranking pairs removed.

