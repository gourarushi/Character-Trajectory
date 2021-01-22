# pattern-relevance

**Type:**  
Student seminar WS20/21

**Topic:**  
Evaluation of Pattern Clustering and Relevance Ranking.

**Description:**  
The amount of existing time-series classification tasks increases every day, and deep learning approaches have shown impressive results among different classification tasks [1]. However, there is a limited amount of deep neural networks used in real-world problems due to the lack of their interpretability. In contrast to the image modalities, it is a very challenging task to understand if a patch covers a pattern that is relevant for the classification. Recently, the image domain introduces an automated process of concept segmentation and rating [2]. However, it is much more challenging to adapt this to the time-series domain. The goal of this project is to get insights into the relevance of these patches.

[1] Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Mining and Knowledge Discovery, 31(3), 606-660.  
[2] Ghorbani, A., Wexler, J., Zou, J. Y., & Kim, B. (2019). Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems (pp. 9277-9286).

==================================================

**Instructions:**
- I will highly appreciate if you use [Boards](https://git.opendfki.de/mercier/pattern-relevance/-/boards) in the repository. Your tasks are already added to **"To Do"**.
- Before you start working on a specific task, please move it to **"Doing"**. And once a task is done, please move it to **"Closed"**.
- As soon as a task is finished, push the code to the repository with self-explanatory details in the commit. Please do **not** push work in progress to the repository.
- Do **not** push data files (i.e. dataset, embeddings, models etc.) to the repository as it is only intended for source files. When you want to share data files, use this [Cloud directory](https://cloud.dfki.de/owncloud/index.php/s/xnb8MssggZmkPH7)
- Document the code along with your implementation because it becomes very difficult to do it at the end.
- Create a **"src"** folder for the code, **"images"** folder for plots, and a **"statistics"** folder for evaluation files. This structure will help you to organize the files within the repository.
- Note down your findings in this readme. Use the sections below that are created for each Milestone. Only summarize the finding. For deatails you can create individual files and link them (e.g. M1_readme.md).

**Dataset sources:**  
Huge Collection of time-series datasets: www.timeseriesclassification.com/  
Most famous time-series dataset archive: www.cs.ucr.edu/~eamonn/time_series_data_2018/  
Snythetic Anomaly Detection dataset: www.bit.ly/2UNk0Lo


**Suggested datasets:**  
- Character Trajectories: Back-projection is very interpretable. (Main focus)
- Synthetic Anomaly Dectection: Classification task is interpretable.
- FordA: Wide spread uni-variate datset.
- Electric Devices: Provides huge discrepancy between the classes.
- Daily and Sport Activites: Real-world data that is complex.



==================================================

# Project plan

## Milestone 1 : [Dealing with patches](https://git.opendfki.de/mercier/pattern-relevance/-/milestones/1)

## Milestone 2 : [Cluster patches](https://git.opendfki.de/mercier/pattern-relevance/-/milestones/2)

## Milestone 3 : [Utilize metadata of patches](https://git.opendfki.de/mercier/pattern-relevance/-/milestones/3)

## Milestone 4 : [Enhanced patch classification network](https://git.opendfki.de/mercier/pattern-relevance/-/milestones/4)
