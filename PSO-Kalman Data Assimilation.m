clc; clear; close all;

% 输入数据（第一列：噪声较大的InSAR，第二列：高精度GNSS）
jianceshuju=readmatrix('形变数据.xlsx','Range','B3:M32');%导入所有GNSS和InSAR监测数据
data1 = jianceshuju(:,1:2);
data2 = jianceshuju(:,3:4);
data3 = jianceshuju(:,5:6);
data4 = jianceshuju(:,7:8);
data5 = jianceshuju(:,9:10);
data6 = jianceshuju(:,11:12);
data=[data1;
    data2;
    data3;
    data4;
    data5;
    data6]
fusion=zeros(30,6);
for i=1:30
data=jianceshuju(i,:);
insar=data(:,[1,3,5,7,9,11])'
gnss=data(:,[2,4,6,8,10,12])'
data=[insar,gnss]
% 数据准备
insar = data(:,1);
gnss = data(:,2);
n = length(gnss);
dt = 1;

% ================= 粒子群优化参数设置 =================
nvars = 4;  % 优化参数数量 [q_pos, q_vel, r_insar, r_gnss]
lb = [0.01, 0.001, 0.5, 0.05];   % 参数下限
ub = [1,     0.5,   5,   0.5];   % 参数上限

options = optimoptions('particleswarm', ...
    'SwarmSize', 50, ...
    'MaxIterations', 100, ...
    'Display', 'iter');

% 执行PSO优化
[best_params, best_mse] = particleswarm(@(x)kf_cost(x, insar, gnss, dt), ...
    nvars, lb, ub, options);

% 显示最优参数
fprintf('\n最优参数：\n');
fprintf('q_pos = %.4f\nq_vel = %.4f\n', best_params(1:2));
fprintf('r_insar = %.4f\nr_gnss = %.4f\n', best_params(3:4));
fprintf('优化MSE = %.4f mm?\n', best_mse);

%使用优化参数执行卡尔曼滤波
% 状态空间模型配置
F = [1 dt; 0 1];                    % 状态转移矩阵
H = [1 0; 1 0];                     % 观测矩阵

% 噪声矩阵配置
Q = diag([best_params(1)^2, best_params(2)^2]);
R = diag([best_params(3)^2, best_params(4)^2]);

% 初始化滤波器
x = [gnss(1); 0];                   % 初始状态
P = diag([0.1, 0.01]);              % 初始协方差

% 滤波过程
fused = zeros(n,1);
velocity = zeros(n,1);

for k = 1:n
    % 预测步骤
    x = F * x;
    P = F * P * F' + Q;
    
    % 更新步骤
    z = [insar(k); gnss(k)];
    y = z - H * x;
    S = H * P * H' + R;
    K = P * H' / S;
    
    x = x + K * y;
    P = (eye(2) - K * H) * P;
    
    fused(k) = x(1);
    velocity(k) = x(2);
    fusion(i,:)=fused';
end
end

%绘图
figure('Position',[100 100 900 600])
subplot(2,1,1);
hold on;
plot(insar, 'b^', 'MarkerSize',6, 'DisplayName','InSAR');
plot(gnss, 'go', 'MarkerSize',6, 'DisplayName','GNSS');
plot(fused, 'r-', 'LineWidth',2, 'DisplayName','优化融合');
title('沉降监测数据融合结果');
xlabel('时间期数');
ylabel('沉降量 (mm)');
legend('Location','best');
grid on;

subplot(2,1,2);
stem(velocity, 'LineWidth',1.5, 'Color',[0.5 0 0.8]);
title('估计沉降速度');
xlabel('时间期数');
ylabel('速度 (mm/期)');
grid on;

% 性能指标计算
mse = mean((fused - gnss).^2);
mae = mean(abs(fused - gnss));
corr_coef = corr(fused, gnss);

fprintf('\n性能指标：\n');
fprintf('均方误差(MSE): %.4f mm?\n', mse);
fprintf('平均绝对误差(MAE): %.4f mm\n', mae);
fprintf('相关系数: %.4f\n', corr_coef);
mse1=(sum(((fused-gnss)').^2))/180
rmse1=sqrt(mse1)
% ================= 优化目标函数 =================

writematrix(fusion,'非优化粒子群卡尔曼横向融合.xlsx','Range','A1:F30')