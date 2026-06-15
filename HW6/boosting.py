from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin
from typing import Iterable
import pandas as pd


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        early_stopping_rounds: int | None = None, # это для себя пометка
        eval_metric: str | None = None,#
        eval_set: tuple[np.ndarray] | None = None, #
        use_best_model: bool = False, #
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        cat_features: Iterable | None = None, 
        subsample: float = 0.3,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds#
        self.eval_metric = eval_metric#
        self.eval_set = eval_set#
        self.use_best_model = use_best_model
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.models : list = []
        self.gammas : list = []

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: -y * self.sigmoid(-y * z)  # Исправила формулу на правильную, как в конспекте лекций.
        self.cat_features = cat_features
        self.stats_dict = defaultdict() #словарь подсчета, ключ - признак, значение - столбец где каждое щначение это среднее таргета по категории, лучше хранить его в атрибутах класса
        self.feature_mean = 0
        self.rng = np.random.default_rng(self.random_state) #вот тут уже 5-я жирная(без фэтшейминга) подсказка от амриканского друга, в общем кратко я никак не использовала рэндом стейт и из-за этого моели были очень нестабильными и я попросила его проанализировать почему так
        self.feature_list = []
        self.goss = goss
        self.goss_k = goss_k


    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # ! YOUR CODE HERE !
        mod = self.base_model_class(**self.base_model_params)
        mod.fit(X, y)
        return mod

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        best_iter = 0
        best_score_auc = -1
        best_score_loss = np.inf
        round_without_improvement = 0 #вот идею этой переменной мне подсказал мой хороший американский друг, она нужна чтобы отсмотреть как раз когда модель начинает жеска переобчаться
        self._cat_fit(X_train, y_train)
        X_train = self._cat_transform(X_train)
        if self.eval_set is not None:
            X_val, y_val = self.eval_set
            X_val = self._cat_transform(X_val)
            self.eval_set = (X_val, y_val)
            val_predictions = np.zeros(X_val.shape[0])
        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        # ! YOUR CODE HERE !    
        for i in estimator_range:
            idx = self._bootstrap(X_train)
            f = self.grad_fn
            #вот тут не очень хитрая штука, типа в ините у нас градиент, а так как нужен антиградиент нужно еще минус сделать
            antigrad = -1 * f(y_train, train_predictions)
            X_boot = X_train[idx]
            features = self._rsm(X_boot)
            X_boot_select = X_boot[:, features]
            antigrad_boot = antigrad[idx]
            
            if self.goss:
                goss_idx, weights = self._goss(antigrad_boot)
                X_boot_select = X_boot_select[goss_idx]
                antigrad_boot = antigrad_boot[goss_idx] * weights

            self.feature_list.append(features)


            model = self.partial_fit(X_boot_select, antigrad_boot)
            new_pred = model.predict(X_train[:, features])
            gamma = self._find_optimal_gamma(y_train, train_predictions, new_pred)
            train_predictions += self.learning_rate * gamma * new_pred

            self.models.append(model)
            self.gammas.append(gamma)

            self.history["train_loss"].append(self.loss_fn(y_train, train_predictions))
            self.history["train_roc_auc"].append(self.score(X_train, y_train)) #мб ошибка потом исправить если не работает 

            if self.eval_set is not None:
                X_val, y_val = self.eval_set
                preds_val = model.predict(X_val[:, features])
                val_predictions += self.learning_rate * gamma * preds_val
                val_loss = self.loss_fn(y_val, val_predictions)
                self.history["valid_loss"].append(val_loss)
                self.history["valid_roc_auc"].append(self.score(X_val, y_val))

            if self.eval_metric == "roc-auc":
                cur_score = self.history["valid_roc_auc"][-1]
                if cur_score > best_score_auc:
                    best_score_auc = cur_score
                    best_iter = i
                    round_without_improvement = 0
                else:
                    round_without_improvement += 1

            elif self.eval_metric == "loss":
                cur_score = self.history["valid_loss"][-1]
                if cur_score < best_score_loss:
                    best_score_loss = cur_score
                    best_iter = i
                    round_without_improvement = 0
                else:
                    round_without_improvement += 1

            if self.early_stopping_rounds is not None and self.early_stopping_rounds <= round_without_improvement:
                print(f"Наша остановочка на {i+1}")
                break
        
        if self.use_best_model is True and self.eval_set is not None:
            self.gammas = self.gammas[:best_iter+1]
            self.models = self.models[:best_iter+1]


        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # ! YOUR CODE HERE !
        preds = np.zeros(X.shape[0])
        X = self._cat_transform(X)
        for m, gamma, f in zip(self.models, self.gammas, self.feature_list):
            X_ = X[:, f]
            preds += gamma * self.learning_rate * m.predict(X_)
        preds = self.sigmoid(preds)
        ans = np.zeros([X.shape[0], 2])
        ans[:, 0] = 1 - preds
        ans[:, 1] =  preds
        return ans

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
    
    
    def plot_history(self, keys: str | Iterable[str]):
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            plt.plot(self.history[key], label=key)

        plt.xlabel("Итерация")

        if "loss" in keys[0]:
            plt.ylabel("loss")
        else:
            plt.ylabel("roc-auc")

        plt.legend()
        plt.grid()
        plt.show()
    
    def _cat_fit(self, X: np.array, y: np.ndarray):
        if self.cat_features is None:
            return 
        X_copy = pd.DataFrame(X).copy()
        
        #из-за того что у нас типа -1 и 1, нужно немного починить с помощью эстайп
        y_target = (y==1).astype(int)
        self.feature_mean = y_target.mean()

        for f in self.cat_features:
            category = X_copy.iloc[:, f].copy()
            sums = pd.Series(y_target).groupby(category).cumsum() - y_target #очень жалко, что пришлось все почти переписывать((((
            cnts = category.groupby(category).cumcount()
            self.stats_dict[f] = (sums / cnts).groupby(category).last() #до этого я считала не совсем правильно, так как не учитывала, что нужно брать среднее по таргету для каждой категории, а не просто среднее по признаку, и из-за этого модель была очень плохой
         #вот это штука, до которой я сама не догадалась, а надо было. из-за того, что на валидации и тесте могут вылезти признаки, которых не было на трейне, то вылетит ошибка, поэтому мы заполняем просто средним по таргету)

    
    def _cat_transform(self, X: np.ndarray):
        if self.cat_features is None:
            return X
        
        X_copy = pd.DataFrame(X).copy()
        for f in self.cat_features:
            X_copy.iloc[:, f] = X_copy.iloc[:, f].map(self.stats_dict[f]).fillna(self.feature_mean)

        return X_copy.to_numpy() #надо чтобы не было проблем 
    
    #я решила сделать для бутстрапа отдельную функцию, не ругайте!!
    def _bootstrap(self, X: np.ndarray):
        n_samples = X.shape[0]

        if self.bootstrap_type == 'Bernoulli':
            n_select = max(1, int(self.subsample * n_samples))
            idxs = self.rng.choice(n_samples, size=n_select,replace=False)

        elif self.bootstrap_type == 'Bayesian':
            weights = (-np.log(self.rng.uniform(0, 1, n_samples))) ** self.bagging_temperature
            weights /= weights.sum()
            idxs = self.rng.choice(n_samples, size=n_samples, p=weights)

        else:
            idxs = np.arange(n_samples)

        return idxs
        
    #тут тоже решила отдельную функцию запихнуть, чтобы не громождать оснлвые функции
    def _rsm(self, X : pd.ndarray):
        n_features = X.shape[1]
        if self.rsm is not None:
            n_select = max(1, int(self.rsm * n_features))
            features = self.rng.choice(n_features, size=n_select, replace=False)
            return features
        else:
            return np.arange(n_features)
    
    def _goss(self, antigrad: np.ndarray):
        n = len(antigrad)
        cnt = int(self.goss_k * n)
        top_idxs = np.argsort(np.abs(antigrad))[-cnt:]
        min_idxs = np.argsort(np.abs(antigrad))[:-cnt]
        n_samples = len(min_idxs)
        n_select = max(1, int(self.subsample * n_samples))
        little_idxs = self.rng.choice(n_samples, size=n_select,replace=False)
        idxs = np.concatenate([top_idxs, min_idxs[little_idxs]])
        weights = np.ones(len(idxs))
        weights[len(top_idxs):] = 1 / self.subsample
        return idxs, weights