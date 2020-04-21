Simulating the 2-line Visual Discrimination Experiments using Firing Rate RNNs
------------------------------------------------------------------------------

[[[TODO]]]

..
    The *py_stringclustering* package provides commands to load and cluster a 
    collection of strings. For example, the following command loads a collection 
    of strings from a file stored at ``path_to_file``:
..
        >>> import py_stringclustering as scl
        >>> df = scl.read_data(path_to_file)
..
    The data is returned in a Pandas DataFrame ``df``, which consists of two 
    columns, one column ``name`` consisting of the input strings, and another 
    column ``id`` consisting of unique IDs assigned to the input strings. This
    DataFrame then can be used to perform blocking, reducing the number of string 
    pairs to compute the string similarity measure for. The following example 
    shows an example of blocking:
..
        >>> import py_stringmatching as sm
        >>> import py_stringsimjoin as ssj
        >>> trigramtok = sm.QgramTokenizer(qval=3)
        >>> blocked_pairs = ssj.jaccard_join(df, df, 'id', 'id', 'name', 'name', trigramtok, 0.3)
..
    ``blocked_pairs`` is a Pandas DataFrame consisting of four columns. The ``_id`` column stores 
    an ID for each blocked string pair. Column s ``l_id`` and ``r_id`` contain the IDs of the 
    strings in each blocked pair, each corresponding a string ID in ``df``. Finally, the 
    ``_sim_score`` column contains the similarity of the string pair used in  the blocking 
    condition.
..
    Next, we calculate the pairwise similarities of string pairs in ``blocked_pairs`` using a 
    (ptentially new) similarity measure and create a similarity matrix to be used by the 
    clustering algorithm, as illustrated in the following example:
..
        >>> jaccsim = sm.Jaccard()
        >>> sim_scores = scl.get_sim_scores(df, blocked_pairs, trigramtok, jaccsim)
        >>> sim_matrix = scl.get_sim_matrix(df, sim_scores)

..
    ``sim_matrix`` is a NumPy matrix containing the similarities of string pairs in ``sim_scores`` 
    and zero for all the other string pairs. Then, we feed this matrix to a clustering algorithm, 
    in this case a hiearchical clustering algorithm, which labels the strings with cluster IDs:
..
         >>> from sklearn.cluster import AgglomerativeClustering
         >>> aggcl = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='complete')
         >>> labels = aggcl.fit_predict(sim_matrix)
..
    Finally, we use the ``labels`` to create the string cluster, as follows:

..         >>> str_clusters = scl.get_clusters(df, labels)
    
Please refer to the API reference for more details.
