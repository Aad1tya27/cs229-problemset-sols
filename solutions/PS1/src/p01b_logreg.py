import numpy as np
import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model = LogisticRegression(eps=1e-5)
    params = model.fit(x=x_train, y=y_train)

    util.plot(x_train, y_train, model.theta, './output/p01b.png')
    
    x_eval , y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x=x_eval)
    boolean_eval = y_eval > 0.5
    boolean_pred = y_pred > 0.5
    correct_pred = 0
    for i in range((y_eval.size)):
        if (y_pred > 0.5)[i] == (y_eval > 0.5)[i]:
            correct_pred+=1 
    print("Accuracy of model is: ", correct_pred*100/y_eval.size)

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # FIRST METHOD
        # m,n = x.shape
        # self.theta= np.zeros((n,1))
        # eps = self.eps


        # def sigmoid(z):
        #     return np.where(z >= 0, 
        #         1 / (1 + np.exp(-z)), 
        #         np.exp(z) / (1 + np.exp(z))
        #     )
        
        # def inv_hessian(x, y, params):
        #     m, n = x.shape
        #     hess = np.zeros((n, n))
        #     hf = np.zeros((m,1))
        #     for i in range(m):
        #         z = np.matmul(x[i, :], params)
        #         hf[i] = sigmoid(z)

        #     for j in range(n):
        #         for k in range(n):
        #             hjk = 0
        #             for i in range(m):
        #                 hjk += x[i, j] * x[i, k] * hf[i] * (1 - hf[i])  # Use hf[i] instead of hf[i,:]
        #             hess[j, k] = -hjk
        #     return np.linalg.inv(hess)

        # def grad(x, y, params):
        #     m, n = x.shape
        #     l_vec = np.zeros((n,1))
        #     hf = np.zeros((m,1))
        #     for i in range(m):
        #         z = np.matmul(x[i, :], params)
        #         hf[i,0] = sigmoid(z)

        #     for j in range(n):
        #         lj = 0
        #         for i in range(m):
        #             lj += x[i, j] * (y[i] - hf[i, 0])  # Use hf[i] instead of hf[i,:]
        #         l_vec[j, 0] = lj  # Use l_vec[j] instead of l_vec[j,:]
        #     return l_vec
        # while True:
        #     delta = np.matmul(inv_hessian(x,y,params=self.theta), grad(x,y,params=self.theta)) 
        #     # print(np.linalg.norm(delta))
        #     if np.linalg.norm(delta) < eps:
        #         break
        #     self.theta -= delta


        #SECOND METHOD
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's method
        h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
        while True:
            # Save old theta
            theta_old = np.copy(self.theta)
            
            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            # print(h_x)
            H = (x.T * h_x * (1 - h_x)).dot(x)
            gradient_J_theta = x.T.dot(h_x - y)

            # Updata theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            # End training
            if np.linalg.norm(self.theta-theta_old, ord=1) < self.eps:
                break
        return self.theta
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        y = 1 / (1 + np.exp(-x.dot(self.theta)))
        return y

main("../data/ds2_train.csv","../data/ds2_valid.csv", "../pred/1b.csv")