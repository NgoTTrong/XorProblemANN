#include <iostream> 
#include <math.h>
using namespace std;
double* *createMatrix(int m,int n){
    double* *a = new double *[m];
    for (int i = 0; i < m;i++){
        a[i] = new double[n];
    }
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            a[i][j] = 0.0;
        }
    }
    return a;
}
void elementWise(double* *x,double* *y,int m,int n, double* *result){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
                result[i][j] = x[i][j]* y[i][j];
        }
    }
};
void multiple(double* *x,double* *y,int m1,int n1,int m2,int n2, double* *result){
    for (int i = 0; i < m1;i++){
        for (int j = 0; j < n2;j++){
            double val = 0.0;
            for (int k = 0; k < n1;k++){
                val += x[i][k]*y[k][j];
            }
            result[i][j] = val;
        }
    }
}

void add(double* *x,double* *y,int m,int n,double* *result){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
                result[i][j] = x[i][j] + y[i][j];
        }
    }
}
void specialAdd(double* *x,double* *y,int m,int n){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
                x[i][j] += y[i][j];
        }
    }
}
void sub(double* *x,double* *y,int m,int n,double* *result){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
                result[i][j] = x[i][j] - y[i][j];
        }
    }
}
void oneSub(double* *y,int m,int n,double* *result){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            result[i][j] = 1 - y[i][j]; 
        }
    }
}
void sigmoid(double* *matrix, int m,int n,double* *result){
    for (int i = 0; i < m;i++){
        for (int j = 0;j < n;j++){ 
            result[i][j] = 1/(1 + exp(-matrix[i][j]));
        }
    }
}
void transpose(double* *matrix,int m,int n,double* * result){
    for (int i = 0; i < n;i++){
        for (int j = 0; j < m;j++){
            result[i][j] = matrix[j][i];
        }
    }
}
void forward(double* *x, double* *w1,double* *b1,int m1,int n1,double* *w2,double* *b2,int m2,int n2,double* *a1,double* *a2){
    
    double* *a = createMatrix(m1,1);
    double* *z1 = createMatrix(m1,1);
    multiple(w1,x,m1,n1,m1,1,a);
    add(a,b1,m1,1,z1);
    sigmoid(z1,m1,1,a1);

    double* *b = createMatrix(m2,1);
    double* *z2 = createMatrix(m2,1);
    multiple(w2,a1,m2,n2,m1,1,b);
    add(b,b2,m2,1,z2);
    sigmoid(z2,m2,1,a2);
}

void backProp(double* *x, double* *w1,double* *b1,int m1,int n1,double* *w2,double* *b2,int m2,int n2,double* *a1,double* *a2,double * *y,double* *tempW1,double* *tempW2,double* *tempB1,double* *tempB2){
    double* *tranA1 = createMatrix(1,m1);
    double* *tranW2 = createMatrix(n2,m2);
    double* *tranX =  createMatrix(1,m1);
    transpose(a1,m1,1,tranA1);
    transpose(w2,m2,n2,tranW2);
    transpose(x,m1,1,tranX);

    sub(a2,y,m2,1,tempB2);
    multiple(tempB2,tranA1,m2,1,1,m1,tempW2);


    double* *head = createMatrix(m1,1);
    double* *temp = createMatrix(m1,1);
    double* *tail = createMatrix(m1,1);
    multiple(tranW2,tempB2,n2,m2,m2,1,head);
    oneSub(a1,m1,1,tail);
    elementWise(head,a1,n2,1,temp);
    elementWise(temp,tail,n2,1,tempB1);
    multiple(tempB1,tranX,n2,1,1,m1,tempW1);
}
void update(double* *w1,double* *b1,int m1,int n1,double* *w2,double* *b2,int m2,int n2,double* *x,int mx,int nx,double* *y,int my,int ny,double learningRate){
    double* *gradientW1 = createMatrix(m1,n1);
    double* *gradientW2 = createMatrix(m2,n2);
    double* *gradientB1 = createMatrix(m1,1);
    double* *gradientB2 = createMatrix(m2,1);
    double temp = 0.0;
    for (int i = 0; i < nx;i++){
        double* *tempX = new double*[mx];
        double* *tempY = new double*[1];
        for (int j = 0; j < mx;j++){
            tempX[j] = new double;
        }
        tempY[0] = new double;
        for (int j = 0; j < mx;j++){
            tempX[j][0] = x[j][i];
        }
        tempY[0][0] = y[0][i];

        double* *a1 = createMatrix(m1,1);
        double* *a2 = createMatrix(m2,1);
        double* *tempW1 = createMatrix(m1,n1);
        double* *tempW2 = createMatrix(m2,n2);
        double* *tempB1 = createMatrix(m1,1);
        double* *tempB2 = createMatrix(m2,1);
        forward(tempX,w1,b1,m1,n1,w2,b2,m2,n2,a1,a2);
        backProp(tempX,w1,b1,m1,n1,w2,b2,m2,n2,a1,a2,tempY,tempW1,tempW2,tempB1,tempB2);
        specialAdd(gradientW1,tempW1,m1,n1);
        specialAdd(gradientW2,tempW2,m2,n2);
        specialAdd(gradientB1,tempB1,m1,1);
        specialAdd(gradientB2,tempB2,m2,1);
        temp -= tempY[0][0]*log(a2[0][0]) + (1 - tempY[0][0])*log(1 - a2[0][0]);
    }
    for (int i = 0; i < m1;i++){
        for (int j = 0; j < n1;j++){
            gradientW1[i][j]/=mx;
            gradientW1[i][j]*=learningRate;
            w1[i][j] -= gradientW1[i][j];
        }
    }
    for (int i = 0; i < m2;i++){
        for (int j = 0; j < n2;j++){
            gradientW2[i][j]/=mx;
            gradientW2[i][j]*=learningRate;
            w2[i][j] -= gradientW2[i][j];
        }
    }
    for (int i = 0; i < m1;i++){
        for (int j = 0; j < 1;j++){
            gradientB1[i][j]/=mx;
            gradientB1[i][j]*=learningRate;
            b1[i][j] -= gradientB1[i][j];
        }
    }
    for (int i = 0; i < m2;i++){
        for (int j = 0; j < 1;j++){
            gradientB2[i][j]/=mx;
            gradientB2[i][j]*=learningRate;
            b2[i][j] -= gradientB2[i][j];
        }
    }
    cout<<temp<<endl;
}
void train(double* *w1,double* *b1,int m1,int n1,double* *w2,double* *b2,int m2,int n2,double* *x,int mx,int nx,double* *y,int my,int ny,long trainingTime,double learningRate){
    for (long i = 0; i < trainingTime;i++){
        update(w1,b1,m1,n1,w2,b2,m2,n2,x,mx,nx,y,my,ny,learningRate);
    }
}
void setRandom(double* *a,int m,int n){
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            a[i][j] = (double) rand() / (RAND_MAX) + 1;
        }
    }
}
void predict(double* *x, double* *w1,double* *b1,int m1,int n1,double* *w2,double* *b2,int m2,int n2){
    double* *a1 = createMatrix(m1,1);
    double* *a = createMatrix(m1,1);
    double* *z1 = createMatrix(m1,1);
    multiple(w1,x,m1,n1,m1,1,a);
    add(a,b1,m1,1,z1);
    sigmoid(z1,m1,1,a1);

    double* *a2 = createMatrix(m2,1);
    double* *b = createMatrix(m2,1);
    double* *z2 = createMatrix(m2,1);
    multiple(w2,a1,m2,n2,m1,1,b);
    add(b,b2,m2,1,z2);
    sigmoid(z2,m2,1,a2);

    cout<<a2[0][0]<<endl;
    if (a2[0][0] >= 0.5){
        cout<<"X: "<<x[0][0]<<" "<<x[1][0]<<endl;
        cout<<"Y: 1"<<endl;
    }else{
        cout<<"X: "<<x[0][0]<<" "<<x[1][0]<<endl;
        cout<<"Y: 0"<<endl;
    }
}
int main(){
    int m1 = 2,n1 = 2;
    int m2 = 1,n2 = 2;
    double* *w1 = createMatrix(m1,n1);
    double* *b1 = createMatrix(m1,1);
    double* *w2 = createMatrix(m2,n2);
    double* *b2 = createMatrix(m2,1);

    int mx = 2,nx = 4;
    int my = 1,ny = 4;
    double* *x = createMatrix(mx,nx);
    double* *y = createMatrix(my,ny);

    setRandom(w1,m1,n1);
    setRandom(w2,m2,n2);
    setRandom(b1,m1,1);
    setRandom(b2,m2,1);

    x[0][0] = 0;
    x[1][0] = 0;
    x[0][1] = 0;
    x[1][1] = 1;
    x[0][2] = 1;
    x[1][2] = 0;
    x[0][3] = 1;
    x[1][3] = 1;

    y[0][0] = 0;
    y[0][1] = 1;
    y[0][2] = 1;
    y[0][3] = 0;

    train(w1,b1,m1,n1,w2,b2,m2,n2,x,mx,nx,y,my,ny,5000,0.1);

    double* * preX = createMatrix(2,1);
    preX[0][0] = 1;
    preX[1][0] = 1;
    cout<<"<----------------------------------->\n";
    predict(preX,w1,b1,m1,n1,w2,b2,m2,n2);
    
    return 0;
}