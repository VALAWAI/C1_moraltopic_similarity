import os
import sys

from flask import Flask, Request, request
from pathos.multiprocessing import cpu_count, ProcessPool
from traceback import format_exc

import numpy as np
from scipy.stats import dirichlet, beta
import matplotlib.pyplot as plt
import ternary
from scipy.spatial import distance


def noisize(
    vect: np.ndarray, 
    noise: float
    ) -> np.ndarray:
    """Generate a new version of vect adding noise

    This function generatse a noisy version of the input vector. 
    Noise follows a Dirichlet distribution. 

    Parameters
    ----------
    vect : np.ndarray
        The input vector to be modified into a noisy version.
    noise : float
        The amount of noise to add. 0 will leave the vector as it is, 10 will make the vector almost uniformly distributed in the simplex.
    Returns
    -------
    np.ndarray
    """
    
    assert noise >= 0

    # for the mode: m = (a - 1)/(a0 - K)
    # a = m ( a0 - K ) + 1 = m K ( a0 / K - 1 ) + 1 = 1 + m K / h 
    # defining 1/h = a0 / K - 1 => a0 = K ( 1 + 1 / h )
    
    alpha_vect = 1+vect*len(vect)/(noise+1e-10) # the small value is to avoid issues when noise is 0
    return dirichlet.rvs(alpha_vect)[0]
    
def moraltopic_similarity(
    user_moral: np.ndarray, 
    user_topic: np.ndarray, 
    item_moral: np.ndarray, 
    item_topic: np.ndarray, 
    moral_weight: float = .5, 
    moral_noise: float = 0,
    topic_noise: float = 0
    ) -> float:
    """Calculate the weighted similarity between an item vector and an user vector. 
    
    The similarity is the weighted average of cosine similarity between moral vectors (weighted with moral_weight) and topic vectors (1 -  moral_weight).
    Both parts of the item vector can be modified adding some noise, controlled respectively by moral_noise and topic_noise. 

    Parameters
    ----------
    user_moral : np.ndarray
        The user moral vector
    user_topic : np.ndarray
        The user topic vector
    item_moral : np.ndarray
        The item moral vector
    item_topic : np.ndarray
        The item topic vector
    moral_weight : float = .5
        The weight of the moral cosine similarity in the result. The weight of the topic similarity is 1 - moral_weight
    moral_noise : float = 0
        The amount of noise to add to the moral vector of the item.
    topic_noise : float = 0
        The amount of noise to add to the topic vector of the item.

    Returns
    -------
    float
    """

    if moral_noise > 0:
#        user_moral = noisize(user_moral,moral_noise)
    # NB: noise only on the ITEMS
        item_moral = noisize(item_moral,moral_noise)
    if topic_noise > 0:
#        user_topic = noisize(user_topic,topic_noise)
    # NB: noise only on the ITEMS
        item_topic = noisize(item_topic,topic_noise)
    moral_similarity = 1 - distance.cosine(user_moral,item_moral)
    topic_similarity = 1 - distance.cosine(user_topic,item_topic)
    return moral_weight*moral_similarity+(1-moral_weight)*topic_similarity

def create_app() -> Flask:
    """Create a Flask app that computes the moral-topic similarity value.

    This C1 component of the VALAWAI architecture computes the moral-topic similarity value of
    given items and users moral-topic vectors, allowing to weight differently the two parts, and to add noise the item vector.

    Parameters
    ----------
    Returns
    -------
    Flask
        A Flask application that can process GET /shapley requests.

    """
    __EXPECTED_TYPES = {
        'user_topic': list, 'user_moral': list, 
        'item_topic': list, 'item_moral': list, 
        'moral_weight': float , 'moral_noise': float , 'topic_noise': float      
    }    
    app = Flask(__name__)

    def __check_request(request: Request):
        if not request.is_json:
            return {"error": "Request must be JSON"}, 415
        input_data = request.get_json()
        if not isinstance(input_data, dict):
            return {"error": f"Params must be passed as a dict"}, 400
        return input_data


    @app.get('/moraltopic_similarity')
    def get_similarity():
        input_data = __check_request(request)

        __NEED_KEYS = ['user_topic', 'user_moral', 'item_topic', 'item_moral']
        __OPT_KEYS = ['moral_weight', 'moral_noise', 'topic_noise']

        # check that the input data has all the necessary keys and they are the
        # correct type
        for k in __NEED_KEYS:
            if not k in input_data.keys():
                return {"error": f"missing necessary param {k}"}, 400
        for k in input_data.keys():
            if not isinstance(input_data[k], __EXPECTED_TYPES[k]):
                return {"error": f"type of param {k} must be {__EXPECTED_TYPES[k]}"}, 400

        # get optional arguments
        kwargs = {}
        for k in __OPT_KEYS:
            try:
                kwargs[k] = input_data[k]
            except KeyError:
                continue

        # compute and return
        try:
            simil = moraltopic_similarity(
                np.array(input_data['user_moral']),
                np.array(input_data['user_topic']),
                np.array(input_data['item_moral']),
                np.array(input_data['item_topic']),
                **kwargs
            )
            return {'moraltopic_similarity': simil}, 200
        except Exception:
            return {"error": format_exc()}, 400
        
    return app