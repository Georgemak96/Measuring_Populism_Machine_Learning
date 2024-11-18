###PAPER MANIFESTO PROJECT

library(manifestoR)
library(tidyverse)
library(readxl)
setwd("")

mp_load_cache(file = "manifesto_cache.RData")
mp_which_corpus_version()

###metadata
mpds <- mp_maindataset(version="2023a")
print(head(names(mpds)))
library(haven)
poppa<- read_dta("party_medians.dta") %>% distinct(cmp_id,.keep_all = T) 
mpds_grouped<- mpds %>% filter(party%in%poppa$cmp_id) %>%group_by(party) %>% distinct(edate,.keep_all = T)%>% filter(edate<="2023-12-31",edate>="2010-01-01")%>%arrange(desc(edate)) %>%
  select(party,edate)%>% ungroup() %>% print(n=400)
min(mpds_grouped$edate)
max(mpds_grouped$edate)

mpds_new <-mpds_grouped %>% left_join(mpds) 
mpds_new <- mpds_new%>% left_join(poppa,by=c("party"="cmp_id"))
mpds_new <- mpds_new %>% filter(!is.na(populism))


###
mpds_new_new<-mpds_new[,str_detect(names(mpds_new), "edate|countryname|party|per.|populism$")]
names(mpds_new_new)

mpds_new_new<- mpds_new_new%>% select(-c(id_perm,pervote,peruncod,personalised,nr_experts,mean_expert_response,party.y,party_id))
mpds_new_new<-mpds_new_new[,!str_detect(names(mpds_new_new), "per..._|per...._|per....")]

mpds_new_new <- mpds_new_new %>% drop_na()



###manifestos (486)
mpds_new_new %>% nrow()


###parties (190)
mpds_new_new %>%distinct(party) %>% nrow()



###countries number (26)
mpds_new_new  %>% distinct(countryname) %>% nrow()
mpds_new_new %>% count(partyname) %>% arrange(desc(n)) %>% print(n=177)

###Table 1
#library(rio)
#table_1<- mpds_new_new %>% select(partyname, countryname,partyabbrev,edate) %>% group_by(partyname,countryname,partyabbrev) %>%
  #summarise(dates=str_c(edate,collapse =", ")) %>% arrange(partyname) %>% export("Table_1_1.xlsx")

library(caret)
set.seed(7)


index<- createDataPartition(mpds_new_new$populism, p=.7,list=F)
train<- mpds_new_new%>% slice(index)
test<- mpds_new_new %>% slice(-index)
train$populism
variables<- train %>% select(-party,-partyabbrev,-countryname,-partyname,-edate) 
variables_test<- test%>% select(-party,-partyabbrev,-countryname,-partyname,-edate) 

###preprocess
prep <- preProcess(variables,method = c("center","scale"))
train_prep<- predict(prep,variables)
test_prep<- predict(prep,variables_test)



###rf_model
trControl<- trainControl(method = "cv", number = 10, allowParallel = TRUE)
set.seed(2)
rf_model<- train(populism~.,data=train_prep,method="rf",trControl=trControl)
predictions_rf<-predict(rf_model,test_prep)
rmse_rf<- (test_prep$populism-predictions_rf)^2%>%mean %>% sqrt()
importance<- varImp(rf_model)
caret:::plot.varImp.train(importance,top=20)

###gbm_model
trControl<- trainControl(method = "cv", number = 10, allowParallel = TRUE)
set.seed(2)
gbm_model<- train(populism~., data = train_prep, method="gbm",trControl=trControl)
predictions_gbm <- predict(gbm_model,test_prep)
summary(gbm_model) %>% slice(1:20)
rmse_gbm<- (test_prep$populism-predictions_gbm)^2 %>% mean %>%sqrt()
trellis.par.set(caretTheme())
ggplot(gbm_model) 

###svmlinear
trControl<- trainControl(method="cv",number = 10,allowParallel = T)
set.seed(2)
svm_linear_model<- train(populism~.,data=train_prep,method="svmLinear",trControl=trControl)
predictions_svm_linear=predict(svm_linear_model,test_prep)
rmse_svm_linear<- (test_prep$populism-predictions_svm_linear)^2 %>% mean %>%sqrt()


###svm polynomial kernel
trControl<- trainControl(method="cv",number = 10,allowParallel = T)
set.seed(2)
svm_poly_model<- train(populism~.,data=train_prep,method="svmPoly",trControl=trControl)
predictions_svm_poly=predict(svm_poly_model,test_prep)
rmse_poly<- (test_prep$populism-predictions_svm_poly)^2 %>% mean %>%sqrt()


###svm radial Kernel
trControl<- trainControl(method="cv",number = 10,allowParallel = T)
set.seed(2)
svm_radial_model<- train(populism~.,data=train_prep,method="svmRadial",trControl=trControl)
predictions_svm_radial=predict(svm_radial_model,test_prep)
rmse_radial<- (test_prep$populism-predictions_svm_radial)^2 %>% mean %>%sqrt()

###tree model
library(tree)
tree_model<- tree(populism~., data=train_prep)
set.seed(2)
tree<- cv.tree(tree_model)
prune.tree <- prune.tree(tree_model , best = 5)
plot(prune.tree)
text(prune.tree , pretty = 0)
predictions_tree<- predict(prune.tree,test_prep)
rmse_tree<- (test_prep$populism-predictions_tree)^2 %>% mean %>%sqrt()

###ridge regression
train_x<- train_prep %>% select(-populism) %>% data.matrix()
train_y<- train_prep %>% select(populism) %>% data.matrix()
test_x<- test_prep %>% select(-populism) %>% data.matrix()
test_y<- test_prep %>% select(populism) %>% data.matrix()

library(glmnet)
set.seed(2)
ridge_model<- cv.glmnet(train_x,train_y,alpha=0,type.measure = "mse", family="gaussian")
plot(ridge_model)
predictions_ridge<- predict(ridge_model,test_x,s=ridge_model$lambda.min)
rmse_ridge<- (test_y-predictions_ridge)^2 %>% mean %>%sqrt()
small.lambda.index <- which(ridge_model$lambda == ridge_model$lambda.min)
small.lambda.betas <- ridge_model$glmnet$beta[,small.lambda.index]
small.lambda.betas[order(-small.lambda.betas)][1:20]
ridge_model$lambda[order(ridge_model$lambda)]

###Table 4
table_4<- data.frame(GBM=rmse_gbm,Ridge=rmse_ridge,"SVM (Polynomial Kernel)"=rmse_poly,
                     "SVM (Radial Kernel)"=rmse_radial,"Random Forest"=rmse_rf,"SVM Linear"=rmse_svm_linear,Tree=rmse_tree) %>% t() %>% data.frame()
table_4<- table_4%>% rename("RMSE"=".")
row.names(table_4)


#table_4 %>% export("Table 4.xslx")
var_importance<- summary(gbm_model)
manifesto_codes<- read_csv("codebook_categories_MPDS2020a.csv") %>% slice(-1)
var_importance<- var_importance %>% mutate(var=rownames(var_importance)) %>% left_join(manifesto_codes, by= c("var"="variable_name"))
ridge_betas<- data.frame(names=names(small.lambda.betas), values=small.lambda.betas)
var_importance<- var_importance %>% left_join(ridge_betas,by=c("var"="names"))
var_importance<- var_importance %>% mutate(values=round(values,2)) %>% mutate(values=if_else(values>0,"+","-"))

###Figure 2
var_importance %>% arrange(desc(rel.inf)) %>% slice(1:20) %>% ggplot(aes(x=rel.inf,y=reorder(title,rel.inf)))+
  geom_col()+
  geom_text(aes(label=values), position = position_stack(vjust = 0.5), 
            color = "white", size = 5)+
  xlab("Variable Importance")+
  ylab("Most Important Coding Categories (Top 20)")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14))

###Figure 1
figure_1_data<- test %>% bind_cols(predictions_gbm) %>% left_join(mpds_new_new) %>%
  mutate(populism_me=10*((`...63`-min(`...63`))/(max(`...63`)-min(`...63`))))
figure_1_data  %>%
  mutate(partyabbrev=if_else(partyname=="Movement of Ecologists - Citizens' Cooperation","KOSP",if_else(partyname=="We can","Podemos",partyabbrev))) %>%
  group_by(partyabbrev) %>% summarise(populism_score=mean(populism_me)) %>%  arrange(desc(populism_score)) %>% slice(1:20) %>% 
  ggplot(aes(x=reorder(partyabbrev,populism_score),y=populism_score))+
  geom_col(fill="blue")+
  coord_flip()+
  xlab("20 Most Populist Parties")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14))


###ches validation
ches<- read_csv("1999-2019_CHES_dataset_means(v3).csv")
ches<- ches %>% filter(year%in%c("2019","2014","2010"))

ches_filtered<- ches%>% group_by(cmp_id) %>%
  summarise(antielite_salience=mean(antielite_salience,na.rm=T),people_vs_elite=mean(people_vs_elite,na.rm=T))
filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$party) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>%
  group_by(test$party) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(ches_filtered,by=c("test$party"="cmp_id"))
party_names<- filtered %>% select("test$party") %>% left_join(mpds_new,by=c("test$party"="party")) %>% rename(party="test$party") %>% distinct(party,.keep_all = T) %>% select(partyabbrev)
library(ggpubr)

### Figure 3
ggplot(filtered,aes(x=antielite_salience,y=populism_me))+
  geom_smooth(method="lm")+
  geom_text(label=party_names$partyabbrev,size=5)+
  stat_cor(method = "pearson", label.x = 1, label.y = 10,size=5)+
  xlab("CHES: Anti-Elite Salience")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14))




####poppa validation (other elements)
poppa_filtered<-poppa %>% group_by(cmp_id) %>% summarise(populism=mean(populism))
filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$party) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>%
  group_by(test$party) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(poppa_filtered,by=c("test$party"="cmp_id"))
cor.test(filtered$populism,filtered$populism_me)

#### ideology
ches_filtered<- ches%>% group_by(cmp_id) %>%
  summarise(lrgen=mean(lrgen,na.rm=T),people_vs_elite=mean(people_vs_elite,na.rm=T), antielite_salience=mean(antielite_salience,na.rm=T)) %>%
  mutate(lrgen=if_else(lrgen<=4,-1,if_else(lrgen>=6,1,0)))

filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$party) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>% 
  group_by(test$party) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(ches_filtered,by=c("test$party"="cmp_id"))%>%
  filter(lrgen %in%c(-1,1,0)) %>%
  mutate(Ideology=factor(lrgen,levels = c(-1,0,1),labels=c("Left","Center","Right")))

party_names<- filtered %>% select("test$party") %>% left_join(mpds_new,by=c("test$party"="party")) %>% rename(party="test$party") %>% distinct(party,.keep_all = T) %>% select(partyabbrev)

###Figure 5
ggplot(filtered,aes(x=antielite_salience,y=populism_me,color=Ideology))+
  geom_smooth(method="lm")+
  stat_cor(method = "pearson", label.x = 1, label.y = 11,size=5)+
  geom_text(label=party_names$partyabbrev,size=5)+
  xlab("CHES: Anti-Elite Salience")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14),strip.text.x = element_text(size=14),legend.text = element_text(size=14))+
  facet_wrap(~Ideology)






####poppa and ideology
ches_filtered<- ches%>% group_by(cmp_id) %>%
  summarise(lrgen=mean(lrgen,na.rm=T))%>% mutate(lrgen=if_else(lrgen<=4,-1,if_else(lrgen>=6,1,0)))

poppa_filtered<-poppa %>% group_by(cmp_id) %>% summarise(manichean=mean(manichean), 
                                                         peoplecentrism=mean(peoplecentrism),generalwill=mean(generalwill),
                                                         indivisble=mean(indivisble),populism=mean(populism),antielitism=mean(antielitism))

filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$party) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>%
  group_by(test$party) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(poppa_filtered,by=c("test$party"="cmp_id")) %>%
  left_join(ches_filtered,by=c("test$party"="cmp_id")) %>% filter(lrgen%in%c(-1,0,1)) %>%
  mutate(Ideology=factor(lrgen,levels = c(-1,0,1),labels=c("Left","Center","Right")))
  

party_names<- filtered %>% select("test$party") %>% left_join(mpds_new,by=c("test$party"="party")) %>% rename(party="test$party") %>% distinct(party,.keep_all = T) %>% select(partyabbrev)

filtered<- filtered %>% pivot_longer(c(manichean, peoplecentrism,indivisble,populism,antielitism,generalwill),names_to = "POPPA",values_to = "value_poppa")

###Figure 6
ggplot(filtered,aes(x=value_poppa,y=populism_me))+
  geom_point()+
  geom_smooth(method="lm")+
  stat_cor(method = "pearson", label.x = 0, label.y = 11,size=4)+
  xlab("POPPA Dimensions")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14),
        strip.text.x = element_text(size=14),strip.text.y = element_text(size=14),legend.text = element_text(size=14))+
  facet_grid(Ideology~POPPA)


###euroscepticism 
ches_filtered<- ches%>% group_by(cmp_id) %>%
  summarise(eu_position=mean(eu_position,na.rm=T))
filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$party) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>% 
  group_by(test$party) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(ches_filtered,by=c("test$party"="cmp_id"))
party_names<- filtered %>% select("test$party") %>% left_join(mpds_new,by=c("test$party"="party")) %>% rename(party="test$party") %>% distinct(party,.keep_all = T) %>% select(partyabbrev)

###Figure 7
filtered %>%mutate(eu_position=8-eu_position)%>% ggplot(aes(x=eu_position,y=populism_me))+
  geom_smooth(method="lm")+
  stat_cor(method = "pearson", label.x = 1, label.y = 10,size=5)+
  geom_text(label=party_names$partyabbrev,size=5)+
  xlab("CHES: Euroscepitism Position")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14))



###VDEM Validation
vdem<- read_csv("V-Dem-CPD-Party-V2.csv")
vdem<- vdem %>% select(v2paenname,v2xpa_popul,v2pashname) %>% group_by(v2paenname) %>% summarise(populism=median(v2xpa_popul)) %>% filter(!is.na(populism))
filtered <- test_prep %>%cbind(test$partyabbrev)%>% cbind(predictions_gbm)%>%cbind(test$partyname) %>%
  mutate(populism_me=10*((predictions_gbm-min(predictions_gbm))/(max(predictions_gbm)-min(predictions_gbm)))) %>% 
  group_by(test$partyname) %>% summarise(populism_me=mean(populism_me))%>%
  left_join(vdem,by=c("test$partyname"="v2paenname"))
party_names<- filtered %>% select("test$partyname") %>% left_join(mpds_new,by=c("test$partyname"="partyname")) %>% rename(party_names="test$partyname") %>% distinct(party_names,.keep_all = T) %>% select(partyabbrev)

###Figure 4
filtered %>% ggplot(aes(x=populism,y=populism_me))+
  geom_smooth(method="lm")+
  stat_cor(method = "pearson", label.x = 0.00, label.y = 10,size=5)+
  geom_text(label=party_names$partyabbrev,size=5)+
  xlab("V-Party: Populism Index")+
  ylab("Populism Score")+
  theme(text = element_text(size = 14),axis.text = element_text(size = 14))

###codes describing 

mp_describe_code("110")

###saving
#mp_save_cache(file = "manifesto_cache.RData")
