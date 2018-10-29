library("gridExtra")
library('ggplot2')

# Steps to Goal---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.steps.hard = c(40679,59647,6921,20834,11374,18121,5213,4475,11488,4911,5747,1849,10137,5810,32493,2414,2754,1093,152,91,72,56,71,56,77,62,63,70,60,49,67,62,66,66,66,66,64,67,70,58,72,54,72,62,63,56,65,62,60,56,64,59,56,68,70,57,74,60,59,66,62,67,58,62,55,67,55,67,61,59,63,73,59,55,57,62,55,58,74,59,64,54,69,57,62,53,74,68,70,56,69,54,69,52,54,72,78,61,57,61)
Policy.Iteration.steps.hard = c(122805,368164,303102,543129,195804,187501,22317,1219719,17767,30040,103995,29657,2894,7826,16692,23126,45931,20473,9898,4457,2791,2139,381,139,74,61,80,62,54,66,75,52,60,64,56,66,69,64,70,63,65,61,64,67,58,65,55,56,66,55,68,66,59,77,77,65,66,64,55,64,52,62,56,68,60,64,81,58,61,58,58,52,66,65,56,63,59,69,67,71,67,72,60,61,68,79,59,56,57,71,56,59,66,72,61,58,69,68,58,53)

Value.Iteration.steps.easy = c(565,75,11,11,14,15,9,18,10,9,10,12,13,13,10,10,9,13,10,13,12,11,13,11,12,9,9,16,13,9,11,14,11,9,17,10,10,14,13,9,13,15,9,15,13,10,13,9,10,14,10,18,12,9,13,13,13,9,9,13,12,11,25,13,10,11,10,15,14,9,10,15,13,10,13,13,15,10,9,9,10,11,12,11,11,9,12,10,9,14,9,11,13,9,17,10,9,16,10,10)
Policy.Iteration.steps.easy = c(1434,14,51,12,11,10,14,13,11,12,9,17,11,9,11,11,11,19,13,9,10,17,14,12,9,17,14,11,14,13,12,12,17,17,9,12,13,12,13,13,10,12,10,16,10,9,9,16,9,15,12,10,9,11,11,9,9,13,22,11,10,13,16,16,10,9,10,10,10,10,9,11,9,10,20,14,11,19,11,11,15,11,19,15,10,12,12,10,12,11,13,12,11,17,16,16,11,13,14,9)

grid.world.steps.Value.Iteration = data.frame(x = c(iterations,iterations),
                                   y = c(Value.Iteration.steps.easy,Value.Iteration.steps.hard),
                                   Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

grid.world.steps.Policy.Iteration = data.frame(x = c(iterations,iterations),
                                              y = c(Policy.Iteration.steps.easy,Policy.Iteration.steps.hard),
                                              Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

p1 <- ggplot(grid.world.steps.Value.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Value Iteration on the two grids')+xlab('Number of Iterations') +
  ylab('Number of Steps to Goal') 
p2 <- ggplot(grid.world.steps.Policy.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Policy Iteration on the two grids')+xlab('Number of Iterations') +
  ylab('Number of Steps to Goal') 

# Time to calculate Policy---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.time.hard = c(134,22,19,31,31,31,34,46,48,41,46,52,56,63,66,71,74,106,97,90,90,91,97,106,109,110,115,119,124,123,132,131,140,139,153,144,155,156,157,168,170,178,177,179,179,193,190,209,201,205,212,214,213,221,219,265,287,230,237,242,262,250,259,256,268,263,266,274,277,277,280,283,288,296,309,326,309,313,318,315,323,323,325,338,337,339,343,349,358,361,355,373,369,370,378,381,378,391,389,394)
Policy.Iteration.time.hard = c(27,16,23,118,159,40,48,56,60,65,177,159,84,127,151,122,177,117,147,146,177,172,161,156,173,180,208,215,190,203,217,218,228,227,255,247,245,248,255,273,269,273,285,290,288,300,310,310,313,326,326,332,339,374,353,360,364,368,377,378,384,407,401,413,410,420,420,442,446,445,450,482,464,461,481,474,486,498,501,496,512,512,527,522,541,533,560,549,573,554,571,582,580,589,604,595,611,616,614,628)

Value.Iteration.time.easy = c(112,3,3,6,16,5,5,5,6,6,6,6,6,29,9,13,42,15,8,20,10,11,10,18,43,24,12,14,17,30,34,33,50,13,12,13,15,14,15,38,18,15,22,15,28,20,33,35,36,32,31,44,28,24,18,20,18,28,42,36,39,41,33,29,54,42,27,44,34,53,62,55,83,61,94,96,59,33,38,43,32,52,52,36,58,40,47,40,83,72,38,40,93,38,66,44,49,57,108,96)
Policy.Iteration.time.easy = c(9,5,15,15,24,12,17,16,13,14,44,17,20,57,23,56,33,50,59,80,19,19,32,57,46,51,61,30,55,65,48,28,28,29,29,33,34,50,49,46,37,50,45,103,59,38,39,63,98,101,53,49,51,100,53,97,105,131,163,244,209,116,61,81,76,117,88,79,124,120,90,103,105,143,137,105,130,93,95,73,114,77,114,123,111,70,85,96,117,126,58,71,56,97,106,92,81,67,91,62)

grid.world.time.Value.Iteration = data.frame(x = c(iterations,iterations),
                                              y = c(Value.Iteration.time.easy,Value.Iteration.time.hard),
                                              Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

grid.world.time.Policy.Iteration = data.frame(x = c(iterations,iterations),
                                               y = c(Policy.Iteration.time.easy,Policy.Iteration.time.hard),
                                               Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

p3 <- ggplot(grid.world.time.Value.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Value Iteration on the two grids') + xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p4 <- ggplot(grid.world.time.Policy.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Policy Iteration on the two grids') + xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')

# Rewards---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.rewards.hard = c(-40577.0,-59545.0,-6819.0,-20732.0,-11272.0,-18019.0,-5111.0,-4373.0,-11386.0,-4809.0,-5645.0,-1747.0,-10035.0,-5708.0,-32391.0,-2312.0,-2652.0,-991.0,-50.0,11.0,30.0,46.0,31.0,46.0,25.0,40.0,39.0,32.0,42.0,53.0,35.0,40.0,36.0,36.0,36.0,36.0,38.0,35.0,32.0,44.0,30.0,48.0,30.0,40.0,39.0,46.0,37.0,40.0,42.0,46.0,38.0,43.0,46.0,34.0,32.0,45.0,28.0,42.0,43.0,36.0,40.0,35.0,44.0,40.0,47.0,35.0,47.0,35.0,41.0,43.0,39.0,29.0,43.0,47.0,45.0,40.0,47.0,44.0,28.0,43.0,38.0,48.0,33.0,45.0,40.0,49.0,28.0,34.0,32.0,46.0,33.0,48.0,33.0,50.0,48.0,30.0,24.0,41.0,45.0,41.0)
Policy.Iteration.rewards.hard = c(-122703.0,-368062.0,-303000.0,-543027.0,-195702.0,-187399.0,-22215.0,-1219617.0,-17665.0,-29938.0,-103893.0,-29555.0,-2792.0,-7724.0,-16590.0,-23024.0,-45829.0,-20371.0,-9796.0,-4355.0,-2689.0,-2037.0,-279.0,-37.0,28.0,41.0,22.0,40.0,48.0,36.0,27.0,50.0,42.0,38.0,46.0,36.0,33.0,38.0,32.0,39.0,37.0,41.0,38.0,35.0,44.0,37.0,47.0,46.0,36.0,47.0,34.0,36.0,43.0,25.0,25.0,37.0,36.0,38.0,47.0,38.0,50.0,40.0,46.0,34.0,42.0,38.0,21.0,44.0,41.0,44.0,44.0,50.0,36.0,37.0,46.0,39.0,43.0,33.0,35.0,31.0,35.0,30.0,42.0,41.0,34.0,23.0,43.0,46.0,45.0,31.0,46.0,43.0,36.0,30.0,41.0,44.0,33.0,34.0,44.0,49.0)

Value.Iteration.rewards.easy = c(-463.0,27.0,91.0,91.0,88.0,87.0,93.0,84.0,92.0,93.0,92.0,90.0,89.0,89.0,92.0,92.0,93.0,89.0,92.0,89.0,90.0,91.0,89.0,91.0,90.0,93.0,93.0,86.0,89.0,93.0,91.0,88.0,91.0,93.0,85.0,92.0,92.0,88.0,89.0,93.0,89.0,87.0,93.0,87.0,89.0,92.0,89.0,93.0,92.0,88.0,92.0,84.0,90.0,93.0,89.0,89.0,89.0,93.0,93.0,89.0,90.0,91.0,77.0,89.0,92.0,91.0,92.0,87.0,88.0,93.0,92.0,87.0,89.0,92.0,89.0,89.0,87.0,92.0,93.0,93.0,92.0,91.0,90.0,91.0,91.0,93.0,90.0,92.0,93.0,88.0,93.0,91.0,89.0,93.0,85.0,92.0,93.0,86.0,92.0,92.0)
Policy.Iteration.rewards.easy = c(-1332.0,88.0,51.0,90.0,91.0,92.0,88.0,89.0,91.0,90.0,93.0,85.0,91.0,93.0,91.0,91.0,91.0,83.0,89.0,93.0,92.0,85.0,88.0,90.0,93.0,85.0,88.0,91.0,88.0,89.0,90.0,90.0,85.0,85.0,93.0,90.0,89.0,90.0,89.0,89.0,92.0,90.0,92.0,86.0,92.0,93.0,93.0,86.0,93.0,87.0,90.0,92.0,93.0,91.0,91.0,93.0,93.0,89.0,80.0,91.0,92.0,89.0,86.0,86.0,92.0,93.0,92.0,92.0,92.0,92.0,93.0,91.0,93.0,92.0,82.0,88.0,91.0,83.0,91.0,91.0,87.0,91.0,83.0,87.0,92.0,90.0,90.0,92.0,90.0,91.0,89.0,90.0,91.0,85.0,86.0,86.0,91.0,89.0,88.0,93.0)

grid.world.Rewards.Value.Iteration = data.frame(x = c(iterations,iterations),
                                             y = c(Value.Iteration.rewards.easy,Value.Iteration.rewards.hard),
                                             Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

grid.world.Rewards.Policy.Iteration = data.frame(x = c(iterations,iterations),
                                              y = c(Policy.Iteration.rewards.easy,Policy.Iteration.rewards.hard),
                                              Grid = c(rep("Easy Grid World",length(iterations)),rep("Hard Grid World",length(iterations))))

p5 <- ggplot(grid.world.Rewards.Value.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Value Iteration on the two grids') + xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy')
p6 <- ggplot(grid.world.Rewards.Policy.Iteration,aes(x=x,y=y,color = Grid))+geom_line()+ggtitle('Policy Iteration on the two grids') + xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy')

grid.arrange(p1,p3,p5,p2,p4,p6,nrow = 2, ncol = 3)
