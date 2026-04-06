%% PCA-RF
%% 加载数据
% 数据格式说明：
% X = [30×14矩阵]
%   第1-6列：6个沉降监测点数据（单位：mm）
%   第7-14列：8个浸润线监测点数据（单位：m）
% y = [30×1标签向量] (2-稳定，1-不稳定)
clc,clear
rng(2023);
a=readmatrix('粒子群卡尔曼融合.xlsx','Range','A1:F30');
b=readmatrix('浸润线.xlsx','Range','M34:T63');
X = [a,b]; % 前6列沉降，后8列浸润线
y = readmatrix('粒子群卡尔曼融合.xlsx','Range','P1:P30');

%% 数据预处理
X_std = zscore(X); 

% 可视化原始数据趋势
figure;
subplot(2,1,1)
plot(X(:,1:6))
title('沉降监测时序变化');
xlabel('监测期数'); ylabel('沉降量(mm)');

subplot(2,1,2)
plot(X(:,7:14))
title('浸润线高程时序变化');
xlabel('监测期数'); ylabel('高程(m)');

%% PCA
[coeff, score, latent, ~, explained] = pca(X_std);

% 方差解释可视化
figure;
pareto(explained);
title('主成分方差贡献率 (累计>95%停止)');

% 自动选择主成分数（累计解释率≥95%）
cumVar = cumsum(explained);
nComp = find(cumVar >= 95, 1);
if isempty(nComp)
    nComp = size(X,2);
end
fprintf('保留前%d个主成分（累计解释率%.2f%%）\n',nComp,cumVar(nComp));

% 主成分得分矩阵
X_pca = score(:,1:nComp);

%% 4. 特征工程可视化
% 主成分载荷矩阵分析
sensorLabels = [...
    arrayfun(@(n)sprintf('Longitudinal deformation%d',n),1:6,'UniformOutput',false),...
    arrayfun(@(n)sprintf('Saturation%d',n),1:8,'UniformOutput',false)];

figure;
heatmap(abs(coeff(:,1:nComp)),...
    'XLabel','Principal component',...
    'YLabel','Original features',...
    'YDisplayLabels',sensorLabels,...
    'ColorLimits',[0 1]);
title('主成分载荷绝对值');

%% 5. 数据集划分,单次划分，70%训练，30%测试
cv = cvpartition(y, 'HoldOut', 0.3);
Xtrain = X_pca(cv.training,:);
Ytrain = y(cv.training);
Xtest = X_pca(cv.test,:);
Ytest = y(cv.test);

%% 6. 随机森林模型参数
model = TreeBagger(50, Xtrain, Ytrain,...
    'Method', 'classification',...
    'OOBPrediction','on',...
    'OOBVarImp','on',...
    'MinLeafSize',5);

%% 7. 模型评估
% 测试集预测

[Ypred, scores] = predict(model, Xtest);
Ypred = str2double(Ypred);

% 性能指标
confMat = confusionmat(Ytest,Ypred);
accuracy = sum(diag(confMat))/sum(confMat(:));
precision = confMat(2,2)/(confMat(2,2)+confMat(1,2));
recall = confMat(2,2)/(confMat(2,2)+confMat(2,1));
f1 = 2*(precision*recall)/(precision+recall);

fprintf('准确率: %.2f%%, F1分数: %.2f\n',accuracy*100,f1);

% 混淆矩阵
figure;
confusionchart(Ytest, Ypred);
title('混淆矩阵');


% ROC曲线
[~,~,~,AUC] = perfcurve(Ytest,scores(:,2),1);
figure;
plotroc(Ytest',scores(:,2)');
title(['ROC曲线 (AUC = ',num2str(AUC),')']);

%% 8. 特征重要性分析
% 主成分重要性
figure;
bar(model.OOBPermutedVarDeltaError);
xlabel('主成分编号'); ylabel('重要性得分');
title('基于主成分的特征重要性');

% 逆向映射到原始特征
compWeights = abs(coeff(:,1:nComp)) * model.OOBPermutedVarDeltaError';
compWeights = compWeights./sum(compWeights);

figure;
bar(compWeights);
set(gca,'XTick',1:14,'XTickLabel',sensorLabels);
xtickangle(45);
ylabel('加权重要性');
title('原始特征综合重要性');

%% LSTM预测的未来数据进行稳定性预测
ab=readmatrix('预测结果.xlsx','Range','A2:F2');
cd=readmatrix('预测结果.xlsx','Range','H2:O2');
new_data=[ab,cd]
new_normalized = (new_data - mean(X))./std(X); % 手动标准化
 new_pca = new_normalized * coeff(:,1:6);
 new_pred = predict(model, new_pca);