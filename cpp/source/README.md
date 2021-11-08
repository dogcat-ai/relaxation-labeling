# relax
Relaxation Labeling, an identification-by-constraints method that works iteratively
# 2021-11-05
## Question: Should the compatibility(i,j,i,j) be set to zero?
Or should it be set 1.0, because object i being label j is completely compatible with object i being label j, but on the other hand, should it be set to zero, since it does not convey any new information.  Another possibility, is to set this compatibility value to some number based on apriori knowledge of the likelihood object i actually being label j.
