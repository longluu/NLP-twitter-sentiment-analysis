

kFold=10;
cvInd=crossvalind('Kfold',size(X,1),kFold);

accuracy=zeros(1,10);
for i=1:10
    foldI=i;

    trainData =X(cvInd~=foldI,:);
    testData  =X(cvInd==foldI,:);
    trainLabel=Y(cvInd~=foldI);
    testLabel =Y(cvInd==foldI);

    svmModel=fitclinear(trainData,trainLabel);
    yhat1=predict(svmModel,testData);


    addpath('./liblinear');
    tic
    [precision,yhat2]=logistic( trainData, ...
        full(trainLabel), testData, full(testLabel));
    toc


    nTrees=100;
    tic
    B=TreeBagger(nTrees,full(trainData),full(trainLabel),...
        'MinLeafSize',1,'Method','classification');
    toc
    testLabelEst=B.predict(full(testData));
    yhat3=str2double(testLabelEst);


    yhatFinal=((yhat1+yhat2+yhat3)/3)>=0.5;

    accuracy(i)=mean(yhatFinal==testLabel);
    
    disp([num2str(i),'th turn'])
end

mean(accuracy)