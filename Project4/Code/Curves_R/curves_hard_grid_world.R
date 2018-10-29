library("gridExtra")
library('ggplot2')

# Hard Grid World ---------------------------------------------------------

# Steps to Goal---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.steps = c(40679,59647,6921,20834,11374,18121,5213,4475,11488,4911,5747,1849,10137,5810,32493,2414,2754,1093,152,91,72,56,71,56,77,62,63,70,60,49,67,62,66,66,66,66,64,67,70,58,72,54,72,62,63,56,65,62,60,56,64,59,56,68,70,57,74,60,59,66,62,67,58,62,55,67,55,67,61,59,63,73,59,55,57,62,55,58,74,59,64,54,69,57,62,53,74,68,70,56,69,54,69,52,54,72,78,61,57,61)
Policy.Iteration.steps = c(122805,368164,303102,543129,195804,187501,22317,1219719,17767,30040,103995,29657,2894,7826,16692,23126,45931,20473,9898,4457,2791,2139,381,139,74,61,80,62,54,66,75,52,60,64,56,66,69,64,70,63,65,61,64,67,58,65,55,56,66,55,68,66,59,77,77,65,66,64,55,64,52,62,56,68,60,64,81,58,61,58,58,52,66,65,56,63,59,69,67,71,67,72,60,61,68,79,59,56,57,71,56,59,66,72,61,58,69,68,58,53)
Q.Learning.steps = c(1827,929,1754,1876,2508,1377,657,414,397,2170,877,309,324,472,985,470,548,177,298,982,975,265,406,254,566,394,465,415,221,290,343,336,225,279,495,231,809,281,250,400,124,686,492,246,256,450,192,630,256,358,416,205,187,349,341,224,179,411,167,140,396,302,189,198,283,330,163,171,134,235,186,177,139,201,101,93,145,138,160,187,190,207,181,130,150,121,158,171,133,208,176,198,108,129,130,87,289,234,116,122)

hard.grid.world.value.iteration.steps = data.frame(x = iterations, y = Value.Iteration.steps)
hard.grid.world.policy.iteration.steps = data.frame(x = iterations, y = Policy.Iteration.steps)
hard.grid.world.Q.Learning.iteration.steps = data.frame(x = iterations, y = Q.Learning.steps)
hard.grid.world.steps = data.frame(x = c(iterations,iterations,iterations),
                                   y = c(Value.Iteration.steps,Policy.Iteration.steps,Q.Learning.steps),
                                   Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p1 <- ggplot(hard.grid.world.value.iteration.steps,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Hard Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Steps to Goal') + scale_y_log10()
p2 <- ggplot(hard.grid.world.policy.iteration.steps,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Hard Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Steps to Goal')+ scale_y_log10()
p3 <- ggplot(hard.grid.world.Q.Learning.iteration.steps,aes(x=x,y=y))+geom_line(color = 'red')+
  ggtitle('Hard Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of Steps to Goal')
p4 <- ggplot(hard.grid.world.steps,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Hard Grid World (Steps to Goal)')+xlab('Number of Iterations') +
  ylab('Number of Steps to Goal') 

p <- grid.arrange(p1, p2, p3, p4, ncol=2,nrow = 2)
#ggsave("hard_grid_world_steps_to_goal.png",width = 10,height = 10)


# Time to calculate Policy---------------------------------------------------------


iterations = seq(1,100)
Value.Iteration.time = c(134,22,19,31,31,31,34,46,48,41,46,52,56,63,66,71,74,106,97,90,90,91,97,106,109,110,115,119,124,123,132,131,140,139,153,144,155,156,157,168,170,178,177,179,179,193,190,209,201,205,212,214,213,221,219,265,287,230,237,242,262,250,259,256,268,263,266,274,277,277,280,283,288,296,309,326,309,313,318,315,323,323,325,338,337,339,343,349,358,361,355,373,369,370,378,381,378,391,389,394)
Policy.Iteration.time = c(27,16,23,118,159,40,48,56,60,65,177,159,84,127,151,122,177,117,147,146,177,172,161,156,173,180,208,215,190,203,217,218,228,227,255,247,245,248,255,273,269,273,285,290,288,300,310,310,313,326,326,332,339,374,353,360,364,368,377,378,384,407,401,413,410,420,420,442,446,445,450,482,464,461,481,474,486,498,501,496,512,512,527,522,541,533,560,549,573,554,571,582,580,589,604,595,611,616,614,628)
Q.Learning.time = c(73,31,72,38,34,32,24,41,50,47,53,42,46,52,48,48,71,52,55,63,65,74,61,65,62,73,77,78,78,81,72,76,72,91,77,87,87,90,76,78,101,95,81,93,96,85,91,105,90,92,102,98,94,105,91,99,114,103,96,110,100,111,109,105,105,111,107,107,100,109,112,111,103,108,118,104,124,99,120,110,114,108,111,125,113,120,125,116,117,117,115,133,124,123,110,117,125,118,132,124)

hard.grid.world.value.time = data.frame(x = iterations, y = Value.Iteration.time)
hard.grid.world.policy.time = data.frame(x = iterations, y = Policy.Iteration.time )
hard.grid.world.Q.Learning.time = data.frame(x = iterations, y = Q.Learning.time)
hard.grid.world.time = data.frame(x = c(iterations,iterations,iterations),
                                  y = c(Value.Iteration.time,Policy.Iteration.time,Q.Learning.time),
                                  Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p5 <- ggplot(hard.grid.world.value.time,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Hard Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p6 <- ggplot(hard.grid.world.policy.time,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Hard Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p7 <- ggplot(hard.grid.world.Q.Learning.time,aes(x=x,y=y))+geom_line(color = 'red')+
  ggtitle('Hard Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p8 <- ggplot(hard.grid.world.time,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Hard Grid World (Time to calculate Policy)')+xlab('Number of Iterations') +
  ylab('Number of Milliseconds to Calculate Policy') 

grid.arrange(p5, p6, p7, p8, ncol=2,nrow = 2)
#ggsave("hard_grid_world_steps_to_goal.png",width = 10,height = 10)

# Rewards---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.rewards = c(-40577.0,-59545.0,-6819.0,-20732.0,-11272.0,-18019.0,-5111.0,-4373.0,-11386.0,-4809.0,-5645.0,-1747.0,-10035.0,-5708.0,-32391.0,-2312.0,-2652.0,-991.0,-50.0,11.0,30.0,46.0,31.0,46.0,25.0,40.0,39.0,32.0,42.0,53.0,35.0,40.0,36.0,36.0,36.0,36.0,38.0,35.0,32.0,44.0,30.0,48.0,30.0,40.0,39.0,46.0,37.0,40.0,42.0,46.0,38.0,43.0,46.0,34.0,32.0,45.0,28.0,42.0,43.0,36.0,40.0,35.0,44.0,40.0,47.0,35.0,47.0,35.0,41.0,43.0,39.0,29.0,43.0,47.0,45.0,40.0,47.0,44.0,28.0,43.0,38.0,48.0,33.0,45.0,40.0,49.0,28.0,34.0,32.0,46.0,33.0,48.0,33.0,50.0,48.0,30.0,24.0,41.0,45.0,41.0)
Policy.Iteration.rewards = c(-122703.0,-368062.0,-303000.0,-543027.0,-195702.0,-187399.0,-22215.0,-1219617.0,-17665.0,-29938.0,-103893.0,-29555.0,-2792.0,-7724.0,-16590.0,-23024.0,-45829.0,-20371.0,-9796.0,-4355.0,-2689.0,-2037.0,-279.0,-37.0,28.0,41.0,22.0,40.0,48.0,36.0,27.0,50.0,42.0,38.0,46.0,36.0,33.0,38.0,32.0,39.0,37.0,41.0,38.0,35.0,44.0,37.0,47.0,46.0,36.0,47.0,34.0,36.0,43.0,25.0,25.0,37.0,36.0,38.0,47.0,38.0,50.0,40.0,46.0,34.0,42.0,38.0,21.0,44.0,41.0,44.0,44.0,50.0,36.0,37.0,46.0,39.0,43.0,33.0,35.0,31.0,35.0,30.0,42.0,41.0,34.0,23.0,43.0,46.0,45.0,31.0,46.0,43.0,36.0,30.0,41.0,44.0,33.0,34.0,44.0,49.0)
Q.Learning.rewards = c(-1725.0,-827.0,-1652.0,-1774.0,-2406.0,-1275.0,-555.0,-312.0,-295.0,-2068.0,-775.0,-207.0,-222.0,-370.0,-883.0,-368.0,-446.0,-75.0,-196.0,-880.0,-873.0,-163.0,-304.0,-152.0,-464.0,-292.0,-363.0,-313.0,-119.0,-188.0,-241.0,-234.0,-123.0,-177.0,-393.0,-129.0,-707.0,-179.0,-148.0,-298.0,-22.0,-584.0,-390.0,-144.0,-154.0,-348.0,-90.0,-528.0,-154.0,-256.0,-314.0,-103.0,-85.0,-247.0,-239.0,-122.0,-77.0,-309.0,-65.0,-38.0,-294.0,-200.0,-87.0,-96.0,-181.0,-228.0,-61.0,-69.0,-32.0,-133.0,-84.0,-75.0,-37.0,-99.0,1.0,9.0,-43.0,-36.0,-58.0,-85.0,-88.0,-105.0,-79.0,-28.0,-48.0,-19.0,-56.0,-69.0,-31.0,-106.0,-74.0,-96.0,-6.0,-27.0,-28.0,15.0,-187.0,-132.0,-14.0,-20.0)

hard.grid.world.value.rewards = data.frame(x = iterations, y = Value.Iteration.rewards)
hard.grid.world.policy.rewards  = data.frame(x = iterations, y = Policy.Iteration.rewards)
hard.grid.world.Q.Learning.rewards  = data.frame(x = iterations, y = Q.Learning.rewards)
hard.grid.world.steps = data.frame(x = c(iterations,iterations,iterations),
                                   y = c(Value.Iteration.rewards,Policy.Iteration.rewards ,Q.Learning.rewards),
                                   Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p9 <- ggplot(hard.grid.world.value.rewards,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Hard Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy') 
p10 <- ggplot(hard.grid.world.policy.rewards,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Hard Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy')
p11 <- ggplot(hard.grid.world.Q.Learning.rewards,aes(x=x,y=y))+geom_line(color = 'red')+
  ggtitle('Hard Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy')
p12 <- ggplot(hard.grid.world.steps,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Hard Grid World (total reward)')+xlab('Number of Iterations') +
  ylab('Number of rewards gained for the optimal policy') 

grid.arrange(p9, p10, p11, p12, ncol=2,nrow = 2)
#ggsave("hard_grid_world_steps_to_goal.png",width = 10,height = 10)
