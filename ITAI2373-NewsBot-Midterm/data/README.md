# Data

The raw dataset is **not** stored in this repository (Kaggle competition rules).

**Source:** [BBC News Classification – Kaggle `learn-ai-bbc`](https://www.kaggle.com/competitions/learn-ai-bbc/data)
**File used:** `BBC News Train.csv` (columns `ArticleId`, `Text`, `Category`; 1,490 rows). `BBC News Test.csv` has no labels and is not used.

## How the notebook gets the data

The notebook downloads the files automatically with the Kaggle API:

1. Accept the competition rules on Kaggle.
2. Create an API token and store it in **Colab → 🔑 Secrets** as `KAGGLE_API_TOKEN`.
3. Run the download cell:

```python
!pip install -q -U "kaggle>=1.8.0"
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
!kaggle competitions download -c learn-ai-bbc
!unzip -o learn-ai-bbc.zip
```

Manual alternative: download `BBC News Train.csv` from Kaggle and upload it to `/content` in Colab.
