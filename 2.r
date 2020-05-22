library(randomForest)
shrooms <- read.csv('agaricus-lepiota.data.txt', header = FALSE)
shrooms$levels<-as.factor(shrooms$V1)
# Quantized Case (p = poisonous, e = edible)
shroomforest<-randomForest(formula=levels~V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23, data=shrooms, type='classification', mtry=5)
# Uncomment and run the following to print the CCM
#shroomforest
