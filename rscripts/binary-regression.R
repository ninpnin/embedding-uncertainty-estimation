library(rjson)
library(rstan)
library(ggplot2)

setwd("~/Work/riksdagen-corpus")



ggplot(df, aes(x=decade, y=accuracy, ymin=0.75, group=supp, fill=supp, color=supp)) + 
  geom_line() +
  geom_point()+
  geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.3,
              position=position_dodge(0.05), colour = NA)
