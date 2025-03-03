# # Project 1: Hurricanes, Climate and Clustering
Please complete this with descriptive information for your group's project.

### [Project Assignment](doc/Proj1_desc.md)

Term: Spring 2025

+ Team #1
+ Team members
    + Sungjoon Park SP
    + Juan Palacios JP
    + Azam Khan AK
    + Andrew Marshall Fagerheim AMF


+ Project summary: In this project, explore connections between hurricane clustering and large-scale, interannual climate conditions. In particular, we investigated whether adding an additional input to the clustering algorithm (power dissipation index - PDI) improves the clusters correlation with ENSO/NAO. During this course of this exploration, we tried using different clustering algorithms (K-Means, DBScan, and Gaussian Mixture Models) and different time intervals of hurricane data.
  
**Contribution statement**: ([more information and examples](doc/a_note_on_contributions.md))

SP, JP, AK, and AMF collaboratively designed the study. AMF worked on comparing clusters using data from different time intervals to determine which data to input to the clustering algorithm. AMF drafted the outline of the data story and all team members added results from their section. AK developed a function to calculate the PDI of a storm and worked on GMM clustering, using the DBI metric to compare the three clustering algorithms implemented. JP worked on the DBSCAN clustering algorithm, standardization and normalization of variables, and contributed to the PDI inclusion in the clustering models and the box plot generation. SP obtained datasets for ENSO, AMM, NAO and conducted boxplot analysis for the clustering results. SP also conducted background research on the three indices and formulated conclusions refarding modulating effects of different indices based on clustering results.
All team members contributed to the GitHub repository and prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is organized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
