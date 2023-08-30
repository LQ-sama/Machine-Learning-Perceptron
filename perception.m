%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % ��������С
center1 = [1,1];        % ��һ����������
center2 = [3,4];        % �ڶ�����������
X = zeros(2*n,100);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(2*n,1);       % ����ǩ
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           %�������ݣ����ĵ�+��˹����
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  ��֪��ģ��   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ѧ��ʵ��,�����֪��ģ�͵Ĳ���(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(100,1);
b = zeros(1);  % ��֪��ģ�� y = x*w + b
ted=0.0000009;%%%�趨���ַ������ԵĲ���
ted1=0.002;
w1=zeros(2,1);%%%�趨���ַ�����w��b
b1=zeros(1);
%%%%%ʹ�ÿα��з���ѵ��ģ��%%%%%%
for j=1:1:99999
for i=1:1:200

if Y(i,:)*(X(i,:)*w1+b1)<=0
w1=w1+ted1*Y(i,:)*X(i,:)';
b1=b1+ted1*Y(i,:);
end
end
end



%%%%%%%% ʹ���ݶ��½���ѵ��ģ�ͣ�����С��f(w,b) = 1/2 * || X*w + ones(2*n,1)*b - Y ||^2
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
%%%%%%%%%%%%%%%%  ����������ͼ  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ������ x*w + b =0 ��ͼ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % �������,length()��ʾ��������
    
x2 = -2 : 0.00001 : 7;
y2 = ( -b1 * ones(1,length(x2)) - w1(1) * x2 )/w1(2); 


% x1Ϊ���������ᣬy1Ϊ����
figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % ���������
plot( x2,y2,'b','LineWidth',1,'MarkerSize',10);  
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');

%%%%%%%%%%%%%%%%%%%  ����  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ���ɲ�������:��ѵ������ͬ�ֲ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 10;                % ������������С
Xt = zeros(2*m,2);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Yt = zeros(2*m,1);       % ����ǩ
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);              % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);      % ���ڶ������ݵ�
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);  

plot( x2,y2,'b','LineWidth',1,'MarkerSize',10); % ���������
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

%%%%%%%%%%%%%%%%%%  ѧ��ʵ��     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ����ģ�͵�Ԥ�����������������ݵ���ʵ����Ƚϣ����������     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

