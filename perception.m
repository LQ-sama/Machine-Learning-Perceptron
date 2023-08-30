%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [3,4];        % 第二类数据中心
X = zeros(2*n,100);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(2*n,1);       % 类别标签
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           %生成数据：中心点+高斯噪声
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % 第一类数据标签为1，第二类为-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  感知机模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  学生实现,求出感知机模型的参数(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(100,1);
b = zeros(1);  % 感知机模型 y = x*w + b
ted=0.0000009;%%%设定两种方法各自的步长
ted1=0.002;
w1=zeros(2,1);%%%设定两种方法的w和b
b1=zeros(1);
%%%%%使用课本中方法训练模型%%%%%%
for j=1:1:99999
for i=1:1:200

if Y(i,:)*(X(i,:)*w1+b1)<=0
w1=w1+ted1*Y(i,:)*X(i,:)';
b1=b1+ted1*Y(i,:);
end
end
end



%%%%%%%% 使用梯度下降法训练模型；即最小化f(w,b) = 1/2 * || X*w + ones(2*n,1)*b - Y ||^2
t1=zeros(100,1);
t2=zeros(1);
for p=1:1:100

for i=1:1:200

for j=1:1:200
t1=t1+X(j,:)'*(X(j,:)*w+b-Y(j,:));
t2=t2+(X(j,:)*w+b-Y(j,:));

end
w=w-ted*t1/200;
b=b-ted*t2/200;
end
end
%%%%%%%%%%%%%%%%  分类器可视图  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 即画出 x*w + b =0 的图像 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % 分类界面,length()表示向量长度
    
x2 = -2 : 0.00001 : 7;
y2 = ( -b1 * ones(1,length(x2)) - w1(1) * x2 )/w1(2); 


% x1为分类界面横轴，y1为纵轴
figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % 画分类界面
plot( x2,y2,'b','LineWidth',1,'MarkerSize',10);  
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');

%%%%%%%%%%%%%%%%%%%  测试  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% 生成测试数据:与训练数据同分布 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 10;                % 测试样本量大小
Xt = zeros(2*m,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Yt = zeros(2*m,1);       % 类别标签
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;        % 第一类数据标签为1，第二类为-1 

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);              % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);      % 画第二类数据点
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);  

plot( x2,y2,'b','LineWidth',1,'MarkerSize',10); % 画分类界面
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

%%%%%%%%%%%%%%%%%%  学生实现     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  给出模型的预测输出，并与测试数据的真实输出比较，计算错误率     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rgh1=0;
rgh2=0;
for re1=1:1:m
xt1=Xt(re1,1);
yt1 = ( -b * ones(1,length(xt1)) - w(1) * xt1 )/w(2);
    if Xt(re1,2)<yt1
        rgh1=rgh1+1;
    end
end
for re1=m+1:1:20
xt1=Xt(re1,1);
yt1 = ( -b * ones(1,length(xt1)) - w(1) * xt1 )/w(2);
if Xt(re1,2)>yt1
    rgh1=rgh1+1;
end
end

for re2=1:1:m
    xt2=Xt(re2,1);
    yt2 = ( -b1 * ones(1,length(xt2)) - w1(1) * xt2 )/w1(2);
if Xt(re2,2)<yt2
    rgh2=rgh2+1;
end
end
for re2=m+1:1:20
     xt2=Xt(re2,1);
    yt2 = ( -b1 * ones(1,length(xt2)) - w1(1) * xt2 )/w1(2);
if Xt(re2,2)>yt2
    rgh2=rgh2+1;
end
end

rgh1=rgh1/20;
rgh2=rgh2/20;
rgh1;
rgh2;

