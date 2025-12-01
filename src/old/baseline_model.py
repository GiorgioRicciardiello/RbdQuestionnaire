import pandas as pd
import numpy as np
from collections import defaultdict
from config.config import config
from itertools import combinations
from sklearn.metrics import roc_auc_score
from collections import namedtuple
from graphviz import Digraph
# ----------------------------
# Traverse the decision tree
# ----------------------------
def get_leaf_path(sample):
    """
    Return a tuple representing the path taken through the tree.
    Each element is the response to q1, q2, q3: values are 0, 1.5, or 1.
    """
    return (sample['q1_rbd'], sample['q2_smell'], sample['q4_constipation'], sample['q5_orthostasis'])



def evaluate_tree(df, label_col='label', threshold=0.5):
    """
    For each leaf:
      - count cases & controls
      - compute purity (risk) = n_cases / n_total
      - compute sensitivity = n_cases / total_cases
      - compute specificity = (total_controls - n_controls) / total_controls
      - classify leaf as 'case' if risk >= threshold else 'control'
    Returns a DataFrame, one row per leaf, with these metrics.
    """
    total_cases = int((df[label_col] == 1).sum())
    total_controls = int((df[label_col] == 0).sum())

    # bucket labels by leaf-path
    leaves = defaultdict(list)
    for _, row in df.iterrows():
        path = get_leaf_path(row)
        leaves[path].append(row[label_col])

    rows = []
    for path, labels in leaves.items():
        arr = np.array(labels)
        n_total = len(arr)
        n_cases = int((arr == 1).sum())
        n_controls = n_total - n_cases
        risk = n_cases / n_total if n_total > 0 else 0.0

        # as classifier at this leaf:
        sens = n_cases / total_cases if total_cases > 0 else np.nan
        spec = (total_controls - n_controls) / total_controls if total_controls > 0 else np.nan

        rows.append({
            'leaf': path,
            'n_total': n_total,
            'n_cases': n_cases,
            'n_controls': n_controls,
            'risk': round(risk, 3),
            'sensitivity': round(sens, 3),
            'specificity': round(spec, 3),
            'classified_as': 'case' if risk >= threshold else 'control'
        })

    return pd.DataFrame(rows)


def plot_tree_with_metrics(df,
                           results_df,
                           label_col:str='label',
                           title="Decision Tree with Node Metrics"):
    """
    Plot the full ternary tree with metrics at each internal node and leaf.
    For internal nodes and leaves, show n_cases, n_controls, and risk.
    df: original DataFrame with question columns and label_col.
    results_df: output of evaluate_tree (leaf-level metrics).
    """
    dot = Digraph(comment=title)
    dot.attr('node', shape='box', style='rounded,filled', fontsize='10')

    # Helper to compute metrics for any subset
    def node_metrics(subdf):
        n_total = len(subdf)
        n_cases = int((subdf[label_col] == 1).sum())
        n_ctrl = n_total - n_cases
        risk = n_cases / n_total if n_total > 0 else 0.0
        return n_total, n_cases, n_ctrl, round(risk, 3)

    # Root node metrics
    root_tot, root_cas, root_ctrl, root_risk = node_metrics(df)
    dot.node("Q1_root", f"Q1: RBD\nN={root_tot}, Cases={root_cas}, Ctrl={root_ctrl}\nRisk={root_risk}",
             fillcolor="#cfe2f3")

    # Level Q1
    for i1 in [0.0, 0.5, 1.0]:
        sub1 = df[df['q1_rbd'] == i1]
        n1, c1, u1, r1 = node_metrics(sub1)
        q2_id = f"Q2_{i1}"
        dot.node(q2_id, f"Q2: Smell\n(Q1={i1})\nN={n1}, Cases={c1}, Ctrl={u1}\nRisk={r1}", fillcolor="#fce5cd")
        dot.edge("Q1_root", q2_id, label=str(i1))

        # Level Q2
        for i2 in [0.0, 0.5, 1.0]:
            sub2 = sub1[sub1['q2_smell'] == i2]
            n2, c2, u2, r2 = node_metrics(sub2)
            q3_id = f"Q3_{i1}_{i2}"
            dot.node(q3_id, f"Q3: Constipation\n(Q1={i1}, Q2={i2})\nN={n2}, Cases={c2}, Ctrl={u2}\nRisk={r2}",
                     fillcolor="#d9ead3")
            dot.edge(q2_id, q3_id, label=str(i2))

            # Level Q3
            for i3 in [0.0, 0.5, 1.0]:
                sub3 = sub2[sub2['q4_constipation'] == i3]
                n3, c3, u3, r3 = node_metrics(sub3)
                q4_id = f"Q4_{i1}_{i2}_{i3}"
                dot.node(q4_id,
                         f"Q4: Orthostasis\n(Q1={i1}, Q2={i2}, Q3={i3})\nN={n3}, Cases={c3}, Ctrl={u3}\nRisk={r3}",
                         fillcolor="#d0e0e3")
                dot.edge(q3_id, q4_id, label=str(i3))

    # Leaf metrics from results_df
    for _, row in results_df.iterrows():
        q1, q2, q3, q4 = row['leaf']
        leaf_id = f"Leaf_{q1}_{q2}_{q3}_{q4}"
        label = (
            f"Leaf\n(Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4})\n"
            f"N={row['n_total']}, Cases={row['n_cases']}, Ctrl={row['n_controls']}\n"
            f"Risk={row['risk']}"
        )
        dot.node(leaf_id, label, shape="note", fillcolor="#fff2cc")
        dot.edge(f"Q4_{q1}_{q2}_{q3}", leaf_id, label=str(q4))

    return dot

if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))

    # %% Cohort selection (questionnaire only, questionnaire and actigrapht)
    df_data = df_data[df_data['has_quest'] == 1]

    # %% decision tree
    # Evaluate the tree
    results_df = evaluate_tree(df_data, label_col='diagnosis', threshold=0.5)
    dot = plot_tree_with_metrics(df_data, results_df, label_col='diagnosis')
    dot.render('tree_with_node_metrics', format='png', view=True)

    results_df_pos = results_df[results_df['risk'] > 0]

    # expand the ‘leaf’ tuples into new columns
    results_df_pos[['q1', 'q2', 'q3', 'q4']] = pd.DataFrame(results_df_pos['leaf'].tolist(),
                                                        index=results_df_pos.index)

    # (optional) drop the original 'leaf' column
    results_df = results_df.drop(columns=['leaf'])

    results_df_pos.rename(columns={'q1': 'RBD Symptoms',
                                   'q2': 'Hyposmia',
                                   'q3': 'Constipation',
                                   'q4': 'Orthostatic  Hypotension'}, inplace=True)

    results_df_pos.sort_values(by=['RBD Symptoms', 'Hyposmia', 'Constipation', 'Orthostatic  Hypotension'], ascending=[False, False, False, False], inplace=True)
    # %%
    # Goal
    # For each combination of questions (e.g., all pairs, triples, etc.):
    # Use those questions to define paths (leaves)
    # Evaluate each leaf as a positive classifier

    def evaluate_subset_binary(df, questions, label_col='label', threshold=0.5):
        """
        Evaluate a screening rule defined by:
          - questions: tuple of column-names to form the tree
          - threshold: minimum leaf purity to call that leaf 'positive'

        Groups samples by leaf, computes:
          - purity per leaf (#cases/#total)
          - positive_leaves = those with purity >= threshold
        Aggregates counts to build:
          - sensitivity = TP/(TP+FN)
          - specificity = TN/(TN+FP)
          - youden_index = sens + spec - 1
          - accuracy = (TP + TN)/N_total
          - risk (PPV) = TP/(TP + FP)

        Returns a 1-row DataFrame with all these metrics.
        """
        # total rows
        total_cases = int((df[label_col] == 1).sum())
        total_controls = int((df[label_col] == 0).sum())
        N_total = total_cases + total_controls

        # bucket by leaf
        leaves = defaultdict(list)
        for _, row in df.iterrows():
            key = tuple(row[q] for q in questions)
            leaves[key].append(row[label_col])

        positive_leaves = set()
        leaf_counts = {}
        for leaf, labs in leaves.items():
            nc = sum(labs)
            nu = len(labs) - nc
            purity = nc / (nc + nu) if (nc + nu) > 0 else 0
            leaf_counts[leaf] = (nc, nu)
            if purity >= threshold:
                positive_leaves.add(leaf)

        # aggregate TP, FP, FN, TN
        TP = sum(nc for leaf, (nc, nu) in leaf_counts.items() if leaf in positive_leaves)
        FP = sum(nu for leaf, (nc, nu) in leaf_counts.items() if leaf in positive_leaves)
        FN = sum(nc for leaf, (nc, nu) in leaf_counts.items() if leaf not in positive_leaves)
        TN = sum(nu for leaf, (nc, nu) in leaf_counts.items() if leaf not in positive_leaves)

        sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        youden = sens + spec - 1
        accuracy = (TP + TN) / N_total if N_total > 0 else np.nan
        risk = TP / (TP + FP) if (TP + FP) > 0 else np.nan  # positive predictive value

        return pd.DataFrame({
            'questions': [questions],
            'threshold': [threshold],
            'sensitivity': [round(sens, 3)],
            'specificity': [round(spec, 3)],
            'youden_index': [round(youden, 3)],
            'accuracy': [round(accuracy, 3)],
            'risk_ppv': [round(risk, 3)],
            'positive_leaves': [positive_leaves]
        })


    def find_best_question_combinations_binary(df, question_cols,
                                               label_col='label',
                                               threshold=0.5,
                                               min_q=1, max_q=None,
                                               sort_by='youden_index'):
        """
        Brute-force all subsets of questions from size min_q..max_q,
        evaluate each, and return a DataFrame sorted by sort_by.
        sort_by can be 'youden_index', 'accuracy', or 'risk_ppv'.
        """
        if max_q is None:
            max_q = len(question_cols)

        rows = []
        for r in range(min_q, max_q + 1):
            for combo in combinations(question_cols, r):
                df_row = evaluate_subset_binary(df, combo,
                                                label_col=label_col,
                                                threshold=threshold)
                rows.append(df_row)

        result = pd.concat(rows, ignore_index=True)
        return result.sort_values(sort_by, ascending=False)


    results_df = find_best_question_combinations_binary(df_data,
                                                        question_cols=[c for c in df_data if c.startswith('q')],
                                                        label_col='diagnosis',
                                                        threshold=0.5)

    # %%
    # a simple struct to hold a split
    Split = namedtuple(
        "Split",
        ["question",  # column name
         "pos_val",  # the response value tested
         "youden",  # J score
         "sens",  # sensitivity at this node
         "spec"]  # specificity at this node
    )

    class GreedyTree:
        def __init__(self, pos_vals=(0, 0.5, 1), diagnosis='label'):
            """
            A greedy decision tree with true N‑ary splits for each response value.


            node_id
            This tells you where in the tree that split was evaluated.
            The root node is "root".
            Every time you split on question Q at value v, you spawn three children named "root_Q_0", "root_Q_0.5", and "root_Q_1" (for the three pos_vals).
            As you recurse, you’ll see names like "root_Q1_1_Q2_0.5", meaning:
            First you split on Q1==1 (that created the branch "root_Q1_1"),
            then within that branch you split on Q2==0.5.
            So by reading node_id, you can trace exactly which sequence of splits leads to that log record.

            question
            This is simply the column name (e.g. "q3") that you tested at that node. Every record is one attempt to split at node_id, using df[question] == pos_val.
            Putting it together, a row like
            node_id	question	pos_val	youden	selected
            root_q1_1_q2_0.5	q5	1	0.42	True
            means:
            We had already split root on q1==1, then took the q2==0.5 branch.
            At that new node, we tried splitting on q5==1, computed Youden = 0.42, and that was the best split among all q* × v∈{0,0.5,1} there—so selected=True.
            Every other row with the same node_id but different question or pos_val are the “losing” candidates at that same spot in the tree.

            Parameters
            ----------
            pos_vals : tuple
                All possible coded responses (e.g. (0, 0.5, 1)).
            diagnosis : str
                Name of the binary label column in the DataFrame (1=case, 0=control).
            """
            self.pos_vals = pos_vals
            self.diagnosis = diagnosis
            self.tree = {}  # node_id -> { leaf: bool, ... }
            self.log_records = []  # list of dicts logging every split attempt

        def _youden_for_split(self, df, q, v):
            """
            Compute sensitivity, specificity, Youden for splitting on df[q] == v.
            Returns a Split record and raw counts for logging.
            """
            mask_pos = df[q] == v
            y = df[self.diagnosis].values
            TP = int(((y == 1) & mask_pos).sum())
            FN = int(((y == 1) & ~mask_pos).sum())
            FP = int(((y == 0) & mask_pos).sum())
            TN = int(((y == 0) & ~mask_pos).sum())
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            youden = sens + spec - 1
            return Split(q, v, youden, sens, spec), TP, FP, TN, FN

        def _make_node(self, df, questions, node_id="root"):
            """
            Recurse: choose best (q, v) split by max Youden, then
            branch into separate children for every v in pos_vals.
            Stop when node is pure or no questions remain.
            """
            N_node = len(df)
            C_node = int((df[self.diagnosis] == 1).sum())
            U_node = N_node - C_node
            purity = C_node / N_node if N_node > 0 else 0

            # Leaf condition: pure or no questions
            if N_node == 0 or purity in (0, 1) or not questions:
                self.tree[node_id] = {
                    'leaf': True,
                    'n_total': N_node,
                    'n_cases': C_node,
                    'n_controls': U_node,
                    'risk': round(purity, 3)
                }
                return

            # Evaluate all candidate splits
            best_youden = -np.inf
            best_split = None
            for q in questions:
                for v in self.pos_vals:
                    split, TP, FP, TN, FN = self._youden_for_split(df, q, v)
                    # Log attempt
                    rec = {
                        'node_id': node_id,
                        'question': q,
                        'pos_val': v,
                        'n_node': N_node,
                        'n_node_cases': C_node,
                        'n_node_controls': U_node,
                        'n_pos': int(mask_pos := (df[q] == v).sum()),
                        'n_pos_cases': TP,
                        'n_pos_controls': mask_pos - TP,
                        'n_neg': N_node - mask_pos,
                        'n_neg_cases': FN,
                        'n_neg_controls': TN,
                        'sensitivity': round(split.sens, 3),
                        'specificity': round(split.spec, 3),
                        'youden': round(split.youden, 3),
                        'selected': False
                    }
                    self.log_records.append(rec)
                    if split.youden > best_youden:
                        best_youden = split.youden
                        best_split = (q, v)

            # Mark selected split in log
            for rec in self.log_records:
                if (rec['node_id'] == node_id and
                        rec['question'] == best_split[0] and
                        rec['pos_val'] == best_split[1] and
                        rec['youden'] == round(best_youden, 3)):
                    rec['selected'] = True
                    break

            # Store split and recurse into *ternary* children
            self.tree[node_id] = {
                'leaf': False,
                'split': Split(best_split[0], best_split[1], best_youden, None, None)
            }
            remaining = [q for q in questions if q != best_split[0]]
            for v in self.pos_vals:
                child_id = f"{node_id}_{best_split[0]}_{v}"
                df_child = df[df[best_split[0]] == v]
                self._make_node(df_child, remaining, child_id)

        def fit(self, df, question_cols):
            """
            Build the full greedy tree with true multi-way splits.
            """
            self.tree.clear()
            self.log_records.clear()
            self._make_node(df, question_cols, node_id="root")

        def get_log_df(self):
            """
            Return a DataFrame logging all split attempts, branches, and metrics.
            """
            return pd.DataFrame(self.log_records)

        def render(self, title="Greedy Screening Tree"):
            """
            Render the fitted tree with one branch per response value.
            """
            dot = Digraph(title)

            def _add(node_id):
                info = self.tree[node_id]
                if info['leaf']:
                    lbl = (f"{node_id}\n"
                           f"N={info['n_total']} Cases={info['n_cases']}\n"
                           f"Risk={info['risk']}")
                    dot.node(node_id, lbl, shape="oval", fillcolor="#ffe699")
                else:
                    sp = info['split']
                    lbl = (f"{sp.question}=={sp.pos_val}?\nJ={sp.youden:.2f}")
                    dot.node(node_id, lbl, shape="box", fillcolor="#c9daf8")
                    for v in self.pos_vals:
                        child_id = f"{node_id}_{sp.question}_{v}"
                        dot.edge(node_id, child_id, label=str(v))
                        _add(child_id)

            _add("root")
            return dot
    # assume df_data is your DataFrame, and its 'label' column is 1=case,0=control
    question_cols = [c for c in df_data.columns if c.startswith('q')]


    tree = GreedyTree(pos_vals=(0, 0.5, 1), diagnosis='diagnosis')
    tree.fit(df_data, question_cols)
    dot = tree.render()
    dot.render(filename='my_tree', format='png', cleanup=True, view=True)

    df_records = pd.DataFrame(tree.log_records)

    # %%
    best_row   = results_df.sort_values('risk', ascending=False).iloc[0]
    best_Qs    = best_row['leaf']   # e.g. ('q1_rbd','q2_smell')

    tree = GreedyTree(pos_vals=(0, 0.5, 1), diagnosis='diagnosis')
    tree.fit(df_data, ['q1_rbd', 'q2_smell'])
    dot = tree.render()
    dot.render(filename='my_tree_two_quest', format='png', cleanup=True, view=True)

    # %% Permutations from 1 questions to all the questions
    import pandas as pd
    import numpy as np
    from itertools import combinations


    def _as_col_list(questions):
        # Normalize to a plain list of column names
        if isinstance(questions, str):
            return [questions]
        return list(questions)



    def evaluate_subset(df, questions, label_col='label', threshold=0.5,
                        return_leaf_table=False):
        """
        Evaluate a screening rule built from a *subset* of questions.
        - Groups subjects by the tuple of answers on `questions`
        - Computes leaf risk = n_cases / n_total
        - Calls a leaf 'positive' if risk >= threshold
        - Aggregates TP/FP/FN/TN over leaves to get dataset-level metrics

        Returns:
            summary_df (1 row): questions, sensitivity, specificity, accuracy, PPV (risk),
                                NPV, Youden, and counts
            leaf_df (optional): per-leaf counts & risk for this subset
        """
        cols = _as_col_list(questions)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        y = df[label_col].astype(int).values

        # map each row to a leaf tuple for these questions
        df2 = df.copy()
        df2['_leaf'] = df2.loc[:, cols].apply(tuple, axis=1)

        # leaf stats
        leaf = (
            df2.groupby('_leaf')[label_col]
            .agg(n_total='size', n_cases='sum')
            .reset_index()
        )
        leaf['n_controls'] = leaf['n_total'] - leaf['n_cases']
        leaf['risk'] = leaf['n_cases'] / leaf['n_total']
        leaf['positive'] = leaf['risk'] >= threshold

        # confusion counts
        TP = int(leaf.loc[leaf['positive'], 'n_cases'].sum())
        FP = int(leaf.loc[leaf['positive'], 'n_controls'].sum())
        FN = int(leaf.loc[~leaf['positive'], 'n_cases'].sum())
        TN = int(leaf.loc[~leaf['positive'], 'n_controls'].sum())

        total_cases = TP + FN
        total_controls = TN + FP
        N = total_cases + total_controls

        sens = TP / (TP + FN) if (TP + FN) else np.nan
        spec = TN / (TN + FP) if (TN + FP) else np.nan
        acc = (TP + TN) / N if N else np.nan
        ppv = TP / (TP + FP) if (TP + FP) else np.nan
        npv = TN / (TN + FN) if (TN + FN) else np.nan
        youd = sens + spec - 1 if (pd.notna(sens) and pd.notna(spec)) else np.nan

        summary = pd.DataFrame([{
            'questions': tuple(cols),
            'k': len(cols),
            'threshold': threshold,
            'total_cases': int(total_cases),
            'total_controls': int(total_controls),
            'n_predicted_pos': int(TP + FP),
            'n_predicted_neg': int(TN + FN),
            'sensitivity': round(sens, 3),
            'specificity': round(spec, 3),
            'accuracy': round(acc, 3),
            'risk_ppv': round(ppv, 3),
            'npv': round(npv, 3),
            'youden_index': round(youd, 3),
        }])

        if return_leaf_table:
            expanded = leaf.copy()
            expanded[cols] = pd.DataFrame(expanded['_leaf'].tolist(), index=expanded.index)
            expanded = expanded.drop(columns=['_leaf'])
            expanded = expanded[cols + ['n_total', 'n_cases', 'n_controls', 'risk', 'positive']]
            return summary, expanded

        return summary


    def evaluate_all_subsets(df, question_cols, label_col='label', threshold=0.5,
                             sort_by='youden_index', return_leaves=False):
        """
        Loop over all subset sizes (1..len(question_cols)) and evaluate each.
        Sort by `sort_by` ('youden_index', 'accuracy', or 'risk_ppv').

        If return_leaves=True, also return a dict mapping subset->leaf_table.
        """
        all_rows = []
        leaves = {} if return_leaves else None

        for r in range(1, len(question_cols) + 1):
            for combo in combinations(question_cols, r):
                if return_leaves:
                    summary, leaf_df = evaluate_subset(df, combo, label_col, threshold, True)
                    leaves[tuple(combo)] = leaf_df
                else:
                    summary = evaluate_subset(df, combo, label_col, threshold, False)
                all_rows.append(summary)

        summary_df = pd.concat(all_rows, ignore_index=True).sort_values(sort_by, ascending=False)
        return (summary_df, leaves) if return_leaves else summary_df


    question_cols = [c for c in df_data if c.startswith('q')]

    # 1) Just the ranking of subsets by Youden (or accuracy / risk_ppv)
    summary_df = evaluate_all_subsets(
        df_data,
        question_cols=question_cols,
        label_col='diagnosis',
        threshold=0.5,
        sort_by='youden_index'
    )
    print(summary_df.head(10))

    # 2) Also get each subset’s per-leaf risk table if you want to inspect patterns
    summary_df, leaf_tables = evaluate_all_subsets(
        df_data,
        question_cols=question_cols,
        label_col='diagnosis',
        threshold=0.5,
        sort_by='youden_index',
        return_leaves=True
    )

    # Example: look at the best subset’s leaves
    best_subset = summary_df.iloc[0]['questions']
    leaf_tables[best_subset].head()

    # concatenate into a single frame
    def concat_leaf_tables(leaf_tables):
        """
        leaf_tables: dict { subset(tuple of columns) -> leaf_df }
        Each leaf_df must contain those subset columns + ['n_total','n_cases','n_controls','risk','positive'].
        Returns one stacked DataFrame with metadata about the subset.
        """
        frames = []
        for subset, leaf_df in leaf_tables.items():
            if leaf_df is None or leaf_df.empty:
                continue

            cols = list(subset)
            tmp = leaf_df.copy()

            # Broadcast tuple so pandas doesn't try to align lengths
            tmp['subset'] = [tuple(cols)] * len(tmp)
            tmp['subset_str'] = " + ".join(cols)
            tmp['k'] = len(cols)
            # human-readable pattern like "q1=1; q2=0.5"
            tmp['pattern'] = tmp.apply(lambda r: "; ".join(f"{c}={r[c]}" for c in cols), axis=1)

            frames.append(tmp)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True, sort=False)

        # Reorder columns nicely
        q_cols = sorted({c for s in leaf_tables.keys() for c in s})
        metric_cols = [c for c in ['n_total', 'n_cases', 'n_controls', 'risk', 'positive'] if c in out.columns]
        meta_cols = ['subset', 'subset_str', 'k', 'pattern']
        ordered = meta_cols + q_cols + metric_cols
        out = out[[c for c in ordered if c in out.columns]]

        return out


    all_leaves = concat_leaf_tables(leaf_tables)

    all_leaves.sort_values(by='risk', ascending=False, inplace=True)

    # %%
    n_cases = df_data[df_data['diagnosis'] == 1].shape[0]
    n_controls = df_data.shape[0] - n_cases

    cases_i = 11
    percent_i = 2

    cases_percent =  (cases_i/n_controls)*100
    control_percent = (percent_i/n_controls)*100

    print(f'Cases: {cases_i/n_cases}: {cases_percent:.2f}%')
    print(f'Controls: {percent_i/n_controls}: {control_percent:.2f}%')



