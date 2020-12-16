package com.hankcs.hanlp.model.hmm;

import junit.framework.TestCase;

import java.util.Arrays;

import static com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModelTest.Feel.cold;
import static com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModelTest.Feel.dizzy;
import static com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModelTest.Feel.normal;
import static com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModelTest.Status.Fever;
import static com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModelTest.Status.Healthy;

public class FirstOrderHiddenMarkovModelTest extends TestCase
{

    /**
     * 隐状态
     */
    enum Status
    {
        Healthy,
        Fever,
    }

    /**
     * 显状态
     */
    enum Feel
    {
        normal,
        cold,
        dizzy,
    }
    /**
     * 初始状态概率矩阵
     */
    static float[] start_probability = new float[]{0.6f, 0.4f};
    /**
     * 状态转移概率矩阵
     */
    static float[][] transition_probability = new float[][]{
        {0.7f, 0.3f},
        {0.4f, 0.6f},
    };
    /**
     * 发射概率矩阵
     */
    static float[][] emission_probability = new float[][]{
        {0.5f, 0.4f, 0.1f},
        {0.1f, 0.3f, 0.6f},
    };
    /**
     * 某个病人的观测序列
     */
    static int[] observations = new int[]{normal.ordinal(), cold.ordinal(), dizzy.ordinal()};

    public void testGenerate() throws Exception
    {
        // generate的代码没有看懂
        FirstOrderHiddenMarkovModel givenModel = new FirstOrderHiddenMarkovModel(start_probability, transition_probability, emission_probability);
        for (int[][] sample : givenModel.generate(3, 5, 2))
        {
            for (int t = 0; t < sample[0].length; t++)
                System.out.printf("%s/%s ", Feel.values()[sample[0][t]], Status.values()[sample[1][t]]);
            System.out.println();
        }
    }

    public void testTrain() throws Exception
    {
        FirstOrderHiddenMarkovModel givenModel = new FirstOrderHiddenMarkovModel(start_probability, transition_probability, emission_probability);
        FirstOrderHiddenMarkovModel trainedModel = new FirstOrderHiddenMarkovModel();
        trainedModel.train(givenModel.generate(3, 10, 100000));
        assertTrue(trainedModel.similar(givenModel));
    }

    public void testPredict() throws Exception
    {
        FirstOrderHiddenMarkovModel model = new FirstOrderHiddenMarkovModel(start_probability, transition_probability, emission_probability);
        evaluateModel(model);
    }

    public void evaluateModel(FirstOrderHiddenMarkovModel model)
    {
        int[] pred = new int[observations.length];
        float prob = (float) Math.exp(model.predict(observations, pred));
        int[] answer = {Healthy.ordinal(), Healthy.ordinal(), Fever.ordinal()};
        assertEquals(Arrays.toString(answer), Arrays.toString(pred));
//        assertEquals("0.01512", String.format("%.5f", prob));
        assertEquals("0.015", String.format("%.3f", prob));

        pred = new int[]{pred[0], pred[1]};
        answer = new int[]{answer[0], answer[1]};
        assertEquals(Arrays.toString(answer), Arrays.toString(pred));

        pred = new int[]{pred[0]};
        answer = new int[]{answer[0]};
        assertEquals(Arrays.toString(answer), Arrays.toString(pred));
//        for (int s : pred)
//        {
//            System.out.print(Status.values()[s] + " ");
//        }
//        System.out.printf(" with highest probability of %.5f\n", prob);
    }
}


/*
笔记:

关于hmm，简单来讲就是统计思想，
首先计算初始状态概率向量，就是计算生成的10000个样本中health和fever那一行（第一行，第零行是隐状态）中index=0的频次，然后经过normalizer获取概率值
对于状态转移概率矩阵和发射概率矩阵来讲，也是一个道理，也是统计频率问题。

那么最终结果就变成：
初始状态概率向量： health有600，fever有400，那么health = 0.6, fever=0.4
状态转移概率矩阵：{0.7, 0.3}, {0.4, 0.6}
发射矩阵：{0.5, 0.4, 0.1}, {0.1, 0.3, 0.6}

获取每一步的概率值以后，那么就是预测问题了，预测的求解过程为获取所有可能中概率值最大的输出，这一步可以使用vertbi算法来实现。


此处引申一下：
这里有两种情况，上面方式是获取所有可能中的最优解（即概率值最大的输出），但是同样还有另外一种可能性，就是把所有可能的输出加和作为预测显性序列
的所有可能隐式序列.

此处可以参考：https://www.zhihu.com/question/20962240

在掷骰子这里，根据已知的结果序列，计算所有可能产生此结果序列所有的隐式序列加和，从而判断有没有出老千。


 */

