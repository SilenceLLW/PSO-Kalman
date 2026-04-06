clc,clear
%  30 期的沉降观测时序数据dat
data = readmatrix('粒子群卡尔曼融合.xlsx','Range','A1:F30'); 
data=data(:,1)%选择某监测点数据进行预测
numTimeStepsTrain = 24; % 定义训练数据的时间步长
numTimeStepsTest = length(data) - numTimeStepsTrain; % 测试数据的时间步长

% 数据归一化及数据划分
[Traindata, inputStats] = mapminmax(data');
Traindata=Traindata'
for i=1:27
    p(i,:)=[Traindata(i,:),Traindata(i+1,:),Traindata(i+2,:)]
end
p=p'
% 划分训练集和测试集
XTrain = p(:,1:23);
YTrain = Traindata(4:26)';
XTest = p(:,24:27);
YTest = data(27:30)';


% 定义 LSTM 网络
numFeatures = 3;
numResponses = 1;
numHiddenUnits = 40;

layers = [
    sequenceInputLayer(numFeatures)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...      
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',400, ...      
    'LearnRateDropFactor',0.15, ...      
    'Verbose',1,  ...  
    'Plots','training-progress');    
% 训练 LSTM 网络
net = trainNetwork(XTrain, YTrain, layers, options);

% 进行预测
YPredNorm = predict(net, XTest);
YPred = mapminmax('reverse', YPredNorm, inputStats);


figure;
hold on
plot(1:length(YTest),YTest,'-og')
plot(1:length(YPred),YPred,'-+b')
legend('原始数据','预测数据')
hold off

rmse=sqrt(mean((YPred-YTest).^2))
% 绘制预测结果
figure;
plot(1:numTimeStepsTrain+1, [XTrain(1); YTrain], 'b', 'DisplayName', 'Training Data');
hold on;
plot(numTimeStepsTrain+1:numTimeStepsTrain+numTimeStepsTest, [YTrain(end); YPred], 'r--', 'DisplayName', 'Predicted Data');
hold off;
xlabel('Time Step');
ylabel('Settlement');
title('LSTM Prediction of Settlement');
legend;

save net