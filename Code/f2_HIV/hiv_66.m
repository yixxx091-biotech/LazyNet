function dN=hiv_66(t,N)
global sigma dt beta di f gamma k epison dp g de h
dN=zeros(19,1);
dN(1)=sigma-dt*N(1)-beta*N(1)*(f(1)*N(3)+f(2)*N(6)+f(3)*N(9)+f(4)*N(12)+f(5)*N(15)+f(6)*N(18));

dN(2)=beta*N(1)*f(1)*N(3)-di*N(2)-gamma*N(2);
dN(3)=gamma*N(2)-dp*N(3)-k*N(3)*(epison(1,1)*N(4)+epison(1,2)*N(7)+epison(1,3)*N(10)+epison(1,4)*N(13)+epison(1,5)*N(16)+epison(1,6)*N(19));
dN(4)=g*N(4)*(epison(1,1)*N(3)+epison(2,1)*N(6)+epison(3,1)*N(9)+epison(4,1)*N(12)+epison(5,1)*N(15)+epison(6,1)*N(18))/(h+(epison(1,1)*N(3)+epison(2,1)*N(6)+epison(3,1)*N(9)+epison(4,1)*N(12)+epison(5,1)*N(15)+epison(6,1)*N(18))+N(4))-de*N(4);

dN(5)=beta*N(1)*f(2)*N(6)-di*N(5)-gamma*N(5);
dN(6)=beta*N(1)*f(2)*N(3)*0.03+gamma*N(5)-dp*N(6)-k*N(6)*(epison(2,1)*N(4)+epison(2,2)*N(7)+epison(2,3)*N(10)+epison(2,4)*N(13)+epison(2,5)*N(16)+epison(2,6)*N(19));
dN(7)=g*N(7)*(epison(1,2)*N(3)+epison(2,2)*N(6)+epison(3,2)*N(9)+epison(4,2)*N(12)+epison(5,2)*N(15)+epison(6,2)*N(18))/(h+(epison(1,2)*N(3)+epison(2,2)*N(6)+epison(3,2)*N(9)+epison(4,2)*N(12)+epison(5,2)*N(15)+epison(6,2)*N(18))+N(7))-de*N(7);

dN(8)=beta*N(1)*f(3)*N(9)-di*N(8)-gamma*N(8);
dN(9)=beta*N(1)*f(3)*N(3)*0.03+gamma*N(8)-dp*N(9)-k*N(9)*(epison(3,1)*N(4)+epison(3,2)*N(7)+epison(3,3)*N(10)+epison(3,4)*N(13)+epison(3,5)*N(16)+epison(3,6)*N(19));
dN(10)=g*N(10)*(epison(1,3)*N(3)+epison(2,3)*N(6)+epison(3,3)*N(9)+epison(4,3)*N(12)+epison(5,3)*N(15)+epison(6,3)*N(18))/(h+(epison(1,3)*N(3)+epison(2,3)*N(6)+epison(3,3)*N(9)+epison(4,3)*N(12)+epison(5,3)*N(15)+epison(6,3)*N(18))+N(10))-de*N(10);

dN(11)=beta*N(1)*f(4)*N(12)-di*N(11)-gamma*N(11);
dN(12)=gamma*N(11)-dp*N(12)-k*N(12)*(epison(4,1)*N(4)+epison(4,2)*N(7)+epison(4,3)*N(10)+epison(4,4)*N(13)+epison(4,5)*N(16)+epison(4,6)*N(19));
dN(13)=g*N(13)*(epison(1,4)*N(3)+epison(2,4)*N(6)+epison(3,4)*N(9)+epison(4,4)*N(12)+epison(5,4)*N(15)+epison(6,4)*N(18))/(h+(epison(1,4)*N(3)+epison(2,4)*N(6)+epison(3,4)*N(9)+epison(4,4)*N(12)+epison(5,4)*N(15)+epison(6,4)*N(18))+N(13))-de*N(13);

dN(14)=beta*N(1)*f(5)*N(15)-di*N(14)-gamma*N(14);
dN(15)=beta*N(1)*f(5)*N(12)*0.03+gamma*N(14)-dp*N(15)-k*N(15)*(epison(5,1)*N(4)+epison(5,2)*N(7)+epison(5,3)*N(10)+epison(5,4)*N(13)+epison(5,5)*N(16)+epison(5,6)*N(19));
dN(16)=g*N(16)*(epison(1,5)*N(3)+epison(2,5)*N(6)+epison(3,5)*N(9)+epison(4,5)*N(12)+epison(5,5)*N(15)+epison(6,5)*N(18))/(h+(epison(1,5)*N(3)+epison(2,5)*N(6)+epison(3,5)*N(9)+epison(4,5)*N(12)+epison(5,5)*N(15)+epison(6,5)*N(18))+N(16))-de*N(16);

dN(17)=beta*N(1)*f(6)*N(18)-di*N(17)-gamma*N(17);
dN(18)=beta*N(1)*f(6)*N(12)*0.03+gamma*N(17)-dp*N(18)-k*N(18)*(epison(6,1)*N(4)+epison(6,2)*N(7)+epison(6,3)*N(10)+epison(6,4)*N(13)+epison(6,5)*N(16)+epison(6,6)*N(19));
dN(19)=g*N(19)*(epison(1,6)*N(3)+epison(2,6)*N(6)+epison(3,6)*N(9)+epison(4,6)*N(12)+epison(5,6)*N(15)+epison(6,6)*N(18))/(h+(epison(1,6)*N(3)+epison(2,6)*N(6)+epison(3,6)*N(9)+epison(4,6)*N(12)+epison(5,6)*N(15)+epison(6,6)*N(18))+N(19))-de*N(19);