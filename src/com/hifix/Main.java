package com.hifix;


import com.sun.deploy.util.StringUtils;
import org.junit.Test;
import sun.reflect.generics.tree.Tree;

import java.io.CharArrayReader;
import java.math.BigInteger;
import java.util.Comparator;
import java.util.TreeMap;
import java.util.*;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        // write your code here


    }

    //统计最大和个数
    public int countLargestGroup(int n) {
        int sum = 0, count = 0, max = 0;
        int[] arrays = new int[46];
        for (int i = 1; i <= n; i++) {
            if (i % 10 == 0) {
                int num = i;
                sum = 0;
                while (num > 0) {
                    sum += num % 10;
                    num /= 10;
                }
            }
            max = Math.max(arrays[i % 10 + sum] += 1, max);
        }
        for (int i : arrays)
            count += i == max ? 1 : 0;
        return count;
    }

    //奇偶排序数组
//快速排序O(n^3)
    public int[] sortArrayByParity(int[] A) {
        if (A.length == 0 || A == null) return null;
        int i = 0;
        int length = A.length;
        int j = length - 1;
        while (i < j) {
            while (i < j && A[j] % 2 != 0) {
                j--;
            }
            while (i < j && A[i] % 2 == 0) {

                i++;

            }
            if (i < j) {
                int remp = A[i];
                A[i] = A[j];
                A[j] = remp;
            }


        }

        return A;

    }


    //输入：mat = [[1,2,3],
//            [4,5,6],
//            [7,8,9]]
//输出：25  0 2 1 1 2 2
//解释：对角线的和为：1 + 5 + 9 + 3 + 7 = 25
//请注意，元素 mat[1][1] = 5 只会被计算一次。
    public int diagonalSum(int[][] mat) {
        int length = mat.length;
        int sum = 0;
        if (length % 2 == 0) {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < mat[i].length; j++) {
                    if (i == j) {
                        sum += mat[i][j];
                    }
                }
            }
            int index = 0;
            for (int i = length - 1; i >= 0; i--) {
                sum += mat[index][i];
                index++;
            }


        } else {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < mat[i].length; j++) {
                    if (i == j) {
                        sum += mat[i][j];
                    }
                }

            }
            int index = 0;
            for (int i = length - 1; i >= 0; i--) {
                sum += mat[index][i];
                index++;
            }
            sum -= mat[length / 2][length / 2];
        }


        return sum;
    }


//452. 用最少数量的箭引爆气球

    public int findMinArrowShots(int[][] points) {
        List<List<Integer>> lists = new ArrayList<>();
// 1 2 6 7 8 10 12 16

        int length = points.length;//3
        int xstart = 0;
        while (xstart < length) {
            List<Integer> list = new ArrayList<>();
            list.add(points[xstart][0]);
            list.add(points[xstart][1]);
            lists.add(list);
            xstart++;
        }
        lists.sort((o1, o2) -> {
            return o1.get(1) - o2.get(1);
        });
        int lists_length = lists.size();
        for (int i = 0; i < lists.size(); i++) {
            int xend1 = lists.get(i).get(1);
            int xstart1 = lists.get(i).get(0);
            for (int j = i + 1; j < lists.size(); ) {
                int xstart2 = lists.get(j).get(0);
                int xend2 = lists.get(j).get(1);
                if (xstart2 <= xend1 && xend2 >= xend1 || xstart2 >= xstart1 && xend2 <= xend1 || xstart2 <= xstart1 && xend2 >= xend1 || xend2 > xstart1 && xstart2 <= xstart1) {
                    lists.remove(j);
                } else {
                    j++;
                }

            }
//    if(lists.get(i).get(0)<xend1){
//        lists.remove(i);
//    }

        }
        return lists.size();

    }


    //4+(N-3)*2
    public String convert(String s, int numRows) {
        StringBuilder stringBuilder = new StringBuilder();
        int temp = (numRows - 3) * 2 + 4;
        int length = s.length();
        int index = 0;

        while (index <= numRows) {
            stringBuilder.append(s.charAt(index));
            while (index < length) {
                stringBuilder.append(s.charAt(index + temp));
                index += temp;
            }
            index++;
        }


        return stringBuilder.toString();
    }

//5492. 分割字符串的方案数

    public int numWays(String s) {
        return 1;
    }


    public int reverse(int x) {
        return Integer.reverse(x);

    }

    //两句话中不常见的单词
    public String[] uncommonFromSentences(String A, String B) {
        String[] str1 = A.split(" ");
        String[] str2 = B.split(" ");
        HashMap<String, Integer> map1 = new HashMap<>();
        HashMap<String, Integer> map2 = new HashMap<>();
        for (String st : str1) {
            map1.put(st, map1.getOrDefault(st, 0) + 1);
        }
        for (String st : str2) {
            map2.put(st, map2.getOrDefault(st, 0) + 1);
        }
        List<String> list = new ArrayList<>();
        map1.forEach((s, integer) -> {
            if (integer == 1) {
                if (map2.get(s) == null) {
                    list.add(s);
                }
            }
        });
        map2.forEach((s, integer) -> {
            if (integer == 1) {
                if (map1.get(s) == null) {
                    list.add(s);
                }
            }
        });

        String[] res = new String[list.size()];
        final int[] index = {0};
        list.forEach(s -> {


            res[index[0]++] = s;

        });
        return res;
    }


    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    //               3
//              / \
//              9  20
//              /  \
//              15   7
//迭代的写法
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> levelOrder = new LinkedList<List<Integer>>();
        if (root == null) {
            return levelOrder;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                TreeNode left = node.left, right = node.right;
                if (left != null) {
                    queue.offer(left);
                }
                if (right != null) {
                    queue.offer(right);
                }
            }
            levelOrder.add(0, level);
        }
        return levelOrder;

    }
//递归写法   （突然想起BFS不适合递归）
//public List<List<Integer>> levelOrderBottom(TreeNode root) {
//
//
//
//
//
//        return null;
//}


//1277. 统计全为 1 的正方形子矩阵

    //dp大法
    //思路：dp[i][j]表示以（i,j）作为右下角能够构成正方形的最大边长
    //dp[i][j]=Math.min(Math.min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1
//边界就是第一行和第一列
    public int countSquares(int[][] matrix) {
        int length = matrix.length;
        int mlength = matrix[0].length;
        int res = 0;
        int[][] dp = new int[length][mlength];
//处理第一列
        for (int i = 0; i < length; i++) {
            res += dp[i][0] = matrix[i][0];
        }
//处理第一列
        for (int i = 0; i < mlength; i++) {
            res += dp[0][i] = matrix[0][i];
        }
        if (matrix[0][0] == 1) res--;
//开始DP第一个点应该从（1,1）开始DP

        for (int i = 1; i < length; i++) {
            for (int j = 1; j < mlength; j++) {
                if (matrix[i][j] == 1) {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    res += dp[i][j];
                }


            }
        }

        return res;
    }

//幂集
    /**
     * public List<List<Integer>> subsets(int[] nums) {
     * List<List<Integer>> res=new ArrayList<>();
     * res.add(new ArrayList<>());
     * for(int n:nums){
     * int length=res.size();
     * for(int i=0;i<length;i++){
     * List<Integer> list=new ArrayList<>();
     * list.addAll(res.get(i));
     * list.add(n);
     * res.add(list);
     * }
     * <p>
     * }
     * return res;
     * }
     */
//幂集回溯版本
// [1,2,3]  ->{
// []
// [1] ,[2],[3].
// [1,2],[1,3],[2,3]
// [1,2,3]
// }
    Set<List<Integer>> set = new HashSet<>();

    public List<List<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        backtrack(nums, 0, new ArrayList<Integer>());
        res.addAll(set);
        return res;
    }

    public void backtrack(int[] nums, int first, List<Integer> pre) {
        set.add(new ArrayList<>(pre));
        for (int i = first; i < nums.length; i++) {
            pre.add(nums[i]);
            backtrack(nums, i + 1, pre);
            pre.remove(pre.size() - 1);
        }
    }


    //是特殊位置的元素，则只需要判断对角线处的元素是否为1就行了，happy!
    public int numSpecial(int[][] mat) {
        boolean flag = true;
        int count = 0;
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (mat[i][j] == 0) {
                    continue;
                } else {
                    for (int k = 0; k < mat[0].length; k++) {
                        if (mat[i][k] == 1 && k != j) {
                            flag = false;
                            break;
                        }
                    }
                    for (int l = 0; l < mat.length; l++) {
                        if (mat[l][j] == 1 && l != i) {
                            flag = false;
                            break;
                        }
                    }

                    if (flag == true) {
                        count++;
                    } else {
                        flag = true;
                    }
                }

            }
        }

        return count;
    }


    //    board =
//            [
//            ['A','B','C','E'],
//            ['S','F','C','S'],
//            ['A','D','E','E']
//            ]
    //单词搜索
    private boolean[][] marked;

    //        x-1,y
    // x,y-1  x,y    x,y+1
    //        x+1,y
//    private int[][] direction = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
//    // 盘面上有多少行
//    private int m;
//    // 盘面上有多少列
//    private int n;
//    private String word;
//    private char[][] board;
//
//    public boolean exist(char[][] board, String word) {
//        m = board.length;
//        if (m == 0) {
//            return false;
//        }
//        n = board[0].length;
//        marked = new boolean[m][n];
//        this.word = word;
//        this.board = board;
//
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                if (dfs(i, j, 0)) {
//                    return true;
//                }
//            }
//        }
//        return false;
//    }
//
//    private boolean dfs(int i, int j, int start) {
//        if (start == word.length() - 1) {
//            return board[i][j] == word.charAt(start);
//        }
//        if (board[i][j] == word.charAt(start)) {
//            marked[i][j] = true;
//            for (int k = 0; k < 4; k++) {
//                int newX = i + direction[k][0];
//                int newY = j + direction[k][1];
//                if (inArea(newX, newY) && !marked[newX][newY]) {
//                    if (dfs(newX, newY, start + 1)) {
//                        return true;
//                    }
//                }
//            }
//            marked[i][j] = false;
//        }
//        return false;
//    }
//
//    private boolean inArea(int x, int y) {
//        return x >= 0 && x < m && y >= 0 && y < n;
//    }
//    board =
//            [
//            ['A','B','C','E'],
//            ['S','F','C','S'],
//            ['A','D','E','E']
//            ]
//
//    给定 word = "ABCCED", 返回 true
//    给定 word = "SEE", 返回 true
//    给定 word = "ABCB", 返回 false

    class Solution {
        public List<List<Integer>> groupThePeople(int[] groupSizes) {

            Map<Integer, List<Integer>> map = new HashMap<>();
            List<List<Integer>> result = new ArrayList<>();
            int index = 0;
            for (int i : groupSizes) {
                if (!map.containsKey(i)) {
                    map.put(i, new ArrayList<>());
                }
                List<Integer> sub = map.get(i);
                sub.add(index++);
                map.put(i, sub);

                if (sub.size() == i) {
                    result.add(new ArrayList<>(sub));
                    sub.clear();
                }
            }
            return result;
        }
    }

//全排列注意回溯剪枝！！！！！
//    public List<List<Integer>> permute(int[] nums) {
//        int len = nums.length;
//        // 使用一个动态数组保存所有可能的全排列
//        List<List<Integer>> res = new ArrayList<>();
//        if (len == 0) {
//            return res;
//        }
//
//        boolean[] used = new boolean[len];
//        List<Integer> path = new ArrayList<>();
//
//        dfs(nums, len, 0, path, used, res);
//        return res;
//    }
//
//    private void dfs(int[] nums, int len, int depth,
//                     List<Integer> path, boolean[] used,
//                     List<List<Integer>> res) {
//        if (depth == len) {
//            res.add(new ArrayList<>(path));
//            return;
//        }
//
//        // 在非叶子结点处，产生不同的分支，这一操作的语义是：在还未选择的数中依次选择一个元素作为下一个位置的元素，这显然得通过一个循环实现。
//        for (int i = 0; i < len; i++) {
//            if (!used[i]) {
//                path.add(nums[i]);
//                used[i] = true;
//
//                dfs(nums, len, depth + 1, path, used, res);
//                // 注意：下面这两行代码发生 「回溯」，回溯发生在从 深层结点 回到 浅层结点 的过程，代码在形式上和递归之前是对称的
//                used[i] = false;
//                path.remove(path.size() - 1);
//            }
//        }
//    }
    //秋季个人赛

//1. 速算机器人

    public int calculate(String s) {
        int x = 1;
        int y = 0;
        for (char i : s.toCharArray()) {
            if (i == 'A') {
                x = 2 * x + y;
            } else if (i == 'B') {
                y = 2 * y + x;
            }
        }

        return x + y;
    }

    //2. 早餐组合  staple = [10,20,5], drinks = [5,5,2], x = 15
    //   st=[5,10,20]  dr=[2,5,5]  2+20  5+20
    //二分 2 4 6
    public int breakfastNumber(int[] staple, int[] drinks, int x) {
        Arrays.sort(staple);
        Arrays.sort(drinks);
        int count = 0;
        for (int i : drinks) {
            int temp = x - i;
            if (temp < 0) continue;
            int r = 0;
            int l = staple.length - 1;
            while (r <= l) {
                int mid = (r + l) / 2;
                if (staple[mid] <= temp) {
                    count += (mid - r) + 1;
                    r = mid + 1;
                }
                if (staple[mid] > temp) {
                    l = mid - 1;
                }
            }

            count %= 1000000007;

        }
        return count % 1000000007;
    }


//3. 秋叶收藏集  dp大法启动冲！！
    //leaves = "ryr" leaves = "rrryyyrryyyrr"  "ryyyrryyyr"  "rrrryyrryr"  "rrrrrrrrrryryr"  "rrryrryyyyrryyyryyyrrrr"

    //    空间O(1)  时间O(n);
    public int minimumOperations(String leaves) {
        int[][] dp = new int[leaves.length()][3];
        for (int i = 0; i < leaves.length(); i++) {
            //红
            if (i < 1) {
                dp[i][0] = (leaves.charAt(i) == 'r' ? 0 : 1);
            } else {
                dp[i][0] = dp[i - 1][0] + (leaves.charAt(i) == 'r' ? 0 : 1);
            }
//红黄
            if (i < 1) {
                dp[i][1] = dp[i][0];
            } else {
                dp[i][1] = Math.min(dp[i - 1][0] + (leaves.charAt(i) == 'y' ? 0 : 1), dp[i - 1][1] + (leaves.charAt(i) == 'y' ? 0 : 1));
            }
//红黄红
            if (i < 2) {
                dp[i][2] = dp[i][1];
            } else {
                dp[i][2] = Math.min(dp[i - 1][1] + (leaves.charAt(i) == 'r' ? 0 : 1), dp[i - 1][2] + (leaves.charAt(i) == 'r' ? 0 : 1));
            }
        }


        return dp[leaves.length() - 1][2];
    }


    public int paintingPlan(int n, int k) {
        if (k < n) {
            return 0;
        }
        int sum = 0;
        if (n == k) {
            sum = 2 * n;
        } else if ((n * n) % k == 0) {
        } else if ((n * n) % k == 1) {
            sum = n * n;
        }
        return sum;
    }

    //k=10 n=5;
    public int keyboard(int k, int n) {
        int res = 0;
        if (k < n) {
            int num1;
            for (int i = 1; i <= k; i++) {
                int temp = (sum(26) / (sum(n - k + 1) * sum(26 - n + k - 1))) * (A(n - k + 1, i));
                res += temp;
            }
        } else {
            int num1;
            num1 = n;
            for (int i = 1; i <= n; i++) {
                int temp = (sum(26) / (sum(num1) * sum(26 - num1))) * (A(num1, num1));
                num1--;
                res += temp;
            }
        }
        return res;
    }

    public int sum(int number) {
        int res = 1;
        for (int i = 1; i <= number; i++) {
            res *= i;
        }
        return res;
    }

    public int A(int num1, int num2) {
        int res = 1;
        for (int i = 1; i <= num2; i++) {
            res *= num1;
            num1--;
        }
        return res;
    }

    public boolean isMagic(int[] target) {
        int index = 1;
        int i;
        for (i = 0; i < target.length; i++) {
            if ((index % 2) == (target[i] % 2)) {
                index++;
                continue;
            } else {
                break;
            }
        }
        if (i == target.length) {
            return false;
        }
        return true;
    }
    // k=1的时候21435也是true
    // [2,4,3,1,5]  [2,1,4,3,5]
    //[5,4,3,2,1]

    //text = "   this   is  a sentence"
    public String reorderSpaces(String text) {
        if (text.length() == 1) return text;
        int count = 0;
        int num = 0;
        boolean flag = false;
        List<StringBuilder> list = new ArrayList<>();
        StringBuilder stringBuilder = new StringBuilder();
        for (char i : text.toCharArray()) {
            if (i == ' ') {
                count++;
                if (i == ' ' && flag == true) {
                    num++;
                    list.add(new StringBuilder(stringBuilder));
                    stringBuilder.delete(0, stringBuilder.length());
                }
                flag = false;
            } else {
                stringBuilder.append(i);
                flag = true;
            }
        }
        if (flag == true) {
            num++;
            list.add(new StringBuilder(stringBuilder));
        }

        boolean flag2 = false;
        int res, res2;
        if (num == 1) {
            res = count;
            res2 = count;
        } else {
            res = count / (num - 1);
            res2 = count % (num - 1);
        }
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < list.size() - 1; i++) {
            str.append(list.get(i));
            for (int j = 0; j < res; j++) {
                str.append(' ');
            }

        }
        str.append(list.get(list.size() - 1));
        while (res2 > 0) {
            str.append(' ');
            res2--;
        }

        return str.toString();
    }


    public int maxUniqueSplit(String s) {
        boolean[] flag = new boolean[s.length()];
        flag[0] = true;
        for (int i = 1; i < s.length(); i++) {

        }

        return 1;
    }


    public int maxProductPath(int[][] grid) {
        int[][][] dp = new int[grid.length][grid[0].length][2];
        //0 最大值
        //1 最小值
        dp[0][0][0] = grid[0][0];
        dp[0][0][1] = grid[0][0];
        //处理第一行
        for (int i = 1; i < grid[0].length; i++) {
            dp[0][i][0] = grid[0][i] * dp[0][i - 1][0];
            dp[0][i][1] = grid[0][i] * dp[0][i - 1][1];
        }
        //处理第一列
        for (int i = 1; i < grid.length; i++) {
            dp[i][0][0] = dp[i - 1][0][0] * grid[i][0];
            dp[i][0][1] = dp[i - 1][0][1] * grid[i][0];
        }

        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[i].length; j++) {
                dp[i][j][0] = Math.max(Math.max(grid[i][j] * dp[i - 1][j][0], grid[i][j] * dp[i - 1][j][1]), Math.max(grid[i][j] * dp[i][j - 1][0], grid[i][j] * dp[i][j - 1][1]));

                dp[i][j][1] = Math.min(Math.min(grid[i][j] * dp[i - 1][j][0], grid[i][j] * dp[i - 1][j][1]), Math.min(grid[i][j] * dp[i][j - 1][0], grid[i][j] * dp[i][j - 1][1]));


            }


        }
        long res1 = dp[grid.length - 1][grid[0].length - 1][0] % 1000000007;
        long res2 = dp[grid.length - 1][grid[0].length - 1][1] % 1000000007;
        long res = Math.max(res1, res2);
        return (int) (res < 0 ? -1 : res);

    }

    public List<List<Integer>> suubsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        //backtrack(0, nums, res, new ArrayList<Integer>());
        return res;

    }


    //子集2
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums); //排序
        getAns(nums, 0, new ArrayList<>(), ans);
        return ans;
    }

    private void getAns(int[] nums, int start, ArrayList<Integer> temp, List<List<Integer>> ans) {
        ans.add(new ArrayList<>(temp));
        for (int i = start; i < nums.length; i++) {
            //和上个数字相等就跳过
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            temp.add(nums[i]);
            getAns(nums, i + 1, temp, ans);
            temp.remove(temp.size() - 1);
        }
    }
//括号生成 DFS版本(自动回溯)

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        // 特判
        if (n == 0) {
            return res;
        }

        dfs("", 0, 0, n, res);
        return res;
    }

    private void dfs(String curStr, int left, int right, int n, List<String> res) {
        if (left == n && right == n) {
            res.add(curStr);
            return;
        }

        // 剪枝
        if (left < right) {
            return;
        }

        if (left < n) {
            dfs(curStr + "(", left + 1, right, n, res);
        }
        if (right < n) {
            dfs(curStr + ")", left, right + 1, n, res);
        }
    }


    //根据中序和后序建立二叉树
    Map<Integer, Integer> map = new HashMap<>();
    int[] inorder;
    int[] postorder;
    int postorder_index = 0;

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        this.inorder = inorder;
        this.postorder = postorder;
        int index = 0;
        for (int i : inorder) {
            map.put(i, index);
            index++;

        }
        postorder_index = postorder.length - 1;
        return help(0, inorder.length - 1);

    }

    public TreeNode help(int left, int right) {
        if (left > right) {
            return null;
        }
        int number = postorder[postorder_index];
        TreeNode node = new TreeNode(number);
        int index = 0;
        index = map.get(number);

        postorder_index--;
        node.right = help(index + 1, right);
        node.left = help(left, index - 1);

        return node;
    }


    //    二叉树的根是数组中的最大元素。
//    左子树是通过数组中最大值左边部分构造出的最大二叉树。
//    右子树是通过数组中最大值右边部分构造出的最大二叉树。
    int nums_index = 0;

    public TreeNode constructMaximumBinaryTree(int[] nums) {

        return help(nums, 0, nums.length - 1);
//1 2 6 5 4
    }

    public TreeNode help(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int index = max_index(nums, left, right);
        int num = nums[index];
        TreeNode node = new TreeNode(num);
        node.left = help(nums, left, index - 1);
        node.right = help(nums, index + 1, right);
        return node;
    }

    public int max_index(int[] temp, int left, int right) {
        int max = Integer.MIN_VALUE;
        for (int i = left; i <= right; i++) {
            if (temp[i] > max) {
                max = temp[i];
                nums_index = i;
            }
        }
        return nums_index;
    }


    //源树与克隆找相同结点
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null) {
            return null;
        }
        if (original == target) {
            return cloned;
        }
        // 递归左子树
        TreeNode res = getTargetCopy(original.left, cloned.left, target);
        // 递归右子树
        TreeNode res2 = getTargetCopy(original.right, cloned.right, target);
        return res == null ? res2 : res;
    }

    public int[] LRU(int[][] operators, int k) {
        // write code here
        List<Integer> res = new ArrayList<>();
        List<HashMap<Integer, Integer>> list = new ArrayList<>();
        HashMap<Integer, Integer> map = null;
        int length = operators.length;
        int rlength = operators[0].length;
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < rlength; j++) {
                int op = operators[i][j];
                if (op == 1) {
                    if (k > 0) {
                        map = new HashMap<>();
                        map.put(operators[i][j + 1], operators[i][j + 2]);
                        list.add(0, map);
                        k--;
                        break;
                    } else {
                        list.remove(list.size() - 1);
                        k++;
                    }
                } else {
                    int num = operators[i][j + 1];
                    final boolean[] flag = {false};
                    final int[] po = {0};
                    list.forEach(integerIntegerHashMap -> {
                        if (integerIntegerHashMap.containsKey(num)) {
                            flag[0] = true;
                            po[0] = integerIntegerHashMap.get(num);
                            list.remove(integerIntegerHashMap);
                        }
                    });
                    if (flag.equals(true)) {
                        res.add(po[0]);
                        map = new HashMap<>();
                        map.put(num, po[0]);
                        list.add(0, map);
                    } else {
                        res.add(-1);

                    }


                }
            }
        }
        int[] re = new int[res.size()];
        Iterator<Integer> it = res.iterator();
        int index = 0;
        while (it.hasNext()) {
            re[index++] = it.next();

        }
        return re;
    }


//    List<List<Integer>> lists=new ArrayList<>();
//    List<Integer> list=new ArrayList<>();
//    public List<List<Integer>> pathSum(TreeNode root, int sum) {
//
//        DFS(root,sum);
//
//    return lists;
//    }
//
//public  void DFS(TreeNode root, int number){
//        if(root==null){
//            return ;
//        }
//        list.add(root.val);
//        number-=root.val;
//
//if(root.left==null&&root.right==null&&number==0){
//    lists.add(new ArrayList<>(list));
//}
//DFS(root.left,number);
//DFS(root.right,number);
//list.remove(list.size()-1);
//}


    //全排列一般两种解法，第一种交换全排列
    List<List<Integer>> lists = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);
        lists.add(new ArrayList<>(list));
        for (int i = 1; i < nums.length; i++) {
            List<Integer> temp = new ArrayList<>();
            temp.add(nums[i]);
            lists.add(new ArrayList<>(temp));
            for (int j = 0; j < lists.size() - 1; j++) {
                List<Integer> temp2 = new ArrayList<>();
                temp2 = lists.get(j);
                temp2.add(nums[i]);
                lists.add(new ArrayList<>(temp2));
            }
        }
        return lists;
    }

    //全排列2
    public List<List<Integer>> permuteUnique(int[] nums) {
        int len = nums.length;
        if (len == 0) return null;
        Arrays.sort(nums);
        List<List<Integer>> lists_2 = new ArrayList<>();

        List<Integer> list = new ArrayList<>();
        boolean[] used = new boolean[len];
        backtrack(nums, 0, len, used, list, lists_2);
        return lists_2;
    }

    public void backtrack(int[] nums, int depth, int len, boolean[] used, List<Integer> temp, List<List<Integer>> lists) {
        if (depth == len) {
            lists.add(new ArrayList<>(temp));
            return;
        }
        for (int i = 0; i < len; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            temp.add(nums[i]);
            backtrack(nums, depth + 1, len, used, temp, lists);
            used[i] = false;
            temp.remove(temp.size() - 1);
        }

    }


    //快乐数这可 太快乐了。
    public boolean isHappy(int n) {
        int count = 1;
        while (n != 1) {
            if (count == 1000) return false;
            int number = n;
            int sum = 0;
            while (number / 10 != 0) {
                sum += (number % 10) * (number % 10);
                number /= 10;
            }
            sum += (number) * (number);
            n = sum;
            count++;
        }

        return true;

    }


    static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    //109. 有序链表转换二叉搜索树
    List<Integer> list = new ArrayList<>();

    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        return build(0, list.size() - 1);

    }

    public TreeNode build(int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = getmid(left, right, list);
        TreeNode root = new TreeNode(list.get(mid));
        root.left = build(left, mid - 1);
        root.right = build(mid + 1, right);
        return root;
    }

    public int getmid(int left, int right, List<Integer> list) {
        int mid_index = ((left + (right - left)) / 2);
        return mid_index;
    }

    //简单模拟
    public int[] finalPrices(int[] prices) {
        int[] res = new int[prices.length];
        int dcount = 0;
        int index = -1;
        int number;
        for (int i = 0; i < prices.length; i++) {
            number = prices[i];
            for (int j = i + 1; j < prices.length; j++) {
                if (prices[j] <= prices[i]) {
                    index = j;
                    break;
                }
                if (index != -1) dcount = prices[index];
                number -= dcount;
                res[i] = number;
            }

        }
        return res;
    }


    List<Integer> list_2 = new ArrayList<>();

    public TreeNode increasingBST(TreeNode root) {
        if (root == null) return null;
        ban(root);
        TreeNode node = new TreeNode(list_2.get(0));

        return bd(list_2, node);
    }

    public void ban(TreeNode root) {
        if (root == null)
            return;
        increasingBST(root.left);
        list_2.add(root.val);
        increasingBST(root.right);

    }

    public TreeNode bd(List<Integer> list, TreeNode root) {
        TreeNode head = root;
        for (int i = 1; i < list.size(); i++) {
            int number = list.get(i);
            root.left = null;
            root.right = new TreeNode(number);
            root = root.right;
        }
        return head;
    }

//1332. 删除回文子序列

    public int removePalindromeSub(String s) {
        int length = s.length();
        if (length == 0) {
            return 0;
        } else {
            for (int i = 0, j = length - 1; i < length && j >= 0; i++, j--) {
                if (s.charAt(i) != s.charAt(j)) {
                    return 2;
                }
            }
        }
        //如果本身就是回文字符串就直接返回1；
        return 1;
    }

    //重塑矩阵
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int num1 = nums.length * nums[0].length;
        int num2 = r * c;
        if (num1 < num2) {
            return nums;
        }
        int[][] res = new int[r][c];
        int hang = 0;
        int lie = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums[0].length; j++) {
                res[hang][lie++] = nums[i][j];
                count++;
                if (count >= c) {
                    hang++;
                    count = 0;
                    lie = 0;
                }
            }
        }
        return res;
    }

    //类似摩尔投票法来解决统计子字符串中相同数量连续的0和1(双指针法？？；
    public int countBinarySubstrings(String s) {
        //000111   111000    0001111000 0001100
        //   3        3      6
        int length = s.length();
        if (length == 0) return 0;
        if (length == 1) return 0;
        int count = 0, num1 = 1, num2 = 0;
        int i = 0;
        for (int j = 1; j < length; j++) {
            char current = s.charAt(i);
            if (s.charAt(j) == current) {
                num1++;
                continue;
            }
            if (num1 == (num2 + 1) && s.charAt(j) != current) {
                num2++;
                count++;
                i = j + 1 - num2;
                num1 = num2;
                num2 = 0;
                continue;
            }
            if (num2 < num1 && s.charAt(j) != current && num1 != (num2 + 1)) {
                num2++;
                count++;
            }
            if (num1 >= num2 && (j + 1) < length && s.charAt(j + 1) == current) {
                i = j + 1 - num2;
                num1 = num2;
                num2 = 0;
                continue;
            }
        }

        return count;
    }


    //496. 下一个更大元素 I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] res = new int[nums1.length];
        int number = -1;
        int index = -1;
        int count = 0;
        for (int i : nums1) {
            for (int j = 0; j < nums2.length; j++) {
                if (i == nums2[j]) {
                    index = j;
                }
                if (nums2[j] > i && j > index && index != -1) {
                    number = nums2[j];
                    break;
                }
            }
            res[count++] = number;
            number = -1;
            index = -1;
        }
        return res;
    }


    //回溯剪枝 组合排列问题
    //排列组合问题如果原问题里包含有重复的元素那么在回溯的时候注意使用标记数组进行标记已经进行过回溯的分支达到剪枝的目的。
    List<List<Integer>> lists_h = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        if (k <= 0 || n < k) {
            return lists_h;
        }
        List<Integer> list = new ArrayList<>();
        backtrack(1, n, list, k);
        return lists_h;
    }

    //回溯采用的集合框架最好采用队列减少删除时间。
    public void backtrack(int first, int length, List<Integer> list, int k) {
        if (list.size() == k) {
            lists_h.add(new ArrayList<>(list));
            return;
        }
        //对每一个元素进行回溯
        for (int i = first; i <= length; i++) {
            list.add(i);
            backtrack(i + 1, length, list, k);
            //进行回溯
            list.remove(list.size() - 1);

        }

    }


    public int findSpecialInteger(int[] arr) {
        double length = arr.length;
        if (length == 1) return arr[0];
        length = length * 0.25;
        int count = 1;
//1,1,1,1,2,2,6,6,6,6,7,10
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] == arr[i]) {
                    count++;
                    if (count > length) {
                        return arr[j];
                    }
                } else {
                    count = 1;
                    i = j;
                    break;
                }
            }

        }
        return 0;
    }

    //1170. 比较字符串最小字母出现频次
    public int[] numSmallerByFrequency(String[] queries, String[] words) {
        int[] num1 = new int[queries.length];
        int[] num2 = new int[words.length];
        int p = 0;
        List<Integer> list = new ArrayList<>();
        int index = 0;
        for (String str : queries) {
            num1[index++] = f(str);
        }
        index = 0;
        for (String str : words) {
            num2[index++] = f(str);
        }
        for (int i = 0; i < num1.length; i++) {
            int number = num1[i];
            for (int j = 0; j < num2.length; j++) {
                if (num2[j] > number) {
                    p++;
                }
            }
            list.add(p);
            p = 0;
        }
        int[] res = new int[list.size()];
        final int[] o = {0};
        list.forEach(integer -> {
            res[o[0]++] = integer;
        });
        return res;
    }

    public int f(String s) {
        char[] temp = s.toCharArray();
        Arrays.sort(temp);
        int count = 1;
        char tem = temp[0];
        for (int j = 1; j < temp.length; j++) {
            if (temp[j] == tem) {
                count++;
            } else {
                break;
            }
        }
        return count;
    }


    //独一无二的数
    public boolean uniqueOccurrences(int[] arr) {
        HashSet<Integer> set = new HashSet<>();
        List<Integer> list = new ArrayList<>();
        Arrays.sort(arr);
        int j = 0;
        int num = arr[0];
        int count = 1;
        for (int i = 0; i < arr.length; i++) {
            for (j = i + 1; j < arr.length; j++) {
                if (arr[j] == arr[i]) {
                    count++;
                } else {

                    break;
                }
            }
            i = j - 1;
            list.add(count);
            count = 1;
        }
        List<Integer> list1 = new ArrayList<>();
        count = 1;
        int p = 0;
        list.sort((o1, o2) -> {
            return o1 - o2;
        });
        for (int i = 0; i < list.size(); i++) {
            for (p = i + 1; p < list.size(); p++) {
                if (list.get(p) == list.get(i)) {
                    count++;
                    return false;
                } else {

                    break;
                }
            }
            i = p - 1;
            list1.add(count);
            count = 1;
        }


        return true;
    }


    //就这？
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int sum = 0;
        int[] max1 = new int[grid[0].length];
        int[] max2 = new int[grid.length];
        int index = 0;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] > max) {
                    max = grid[i][j];
                }
            }

            max1[index++] = max;
            max = 0;
        }
        max = 0;
        index = 0;
        for (int i = 0; i < grid[0].length; i++) {
            for (int j = 0; j < grid.length; j++) {
                if (grid[j][i] > max) {
                    max = grid[j][i];
                }
            }
            max2[index++] = max;
            max = 0;
        }
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                sum += Math.min(max1[i], max2[j]) - grid[i][j];

            }
        }
        return sum;
    }


    //特定深度节点链表
//特定深度节点链表
    public static ListNode[] listOfDepth(TreeNode tree) {
        if (tree == null) return null;
        List<ListNode> resArr = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(tree);
        while (!queue.isEmpty()) {
            int size = queue.size();
            ListNode node = new ListNode(0);//头节点
            ListNode head = node;
            for (int i = 0; i < size; i++) {
                TreeNode p = queue.poll();
                ListNode n = new ListNode(p.val);
                node.next = n;
                node = n;
                if (p.left != null) queue.add(p.left);
                if (p.right != null) queue.add(p.right);
            }
            resArr.add(head.next);
        }
        resArr.toArray();
        return resArr.toArray(new ListNode[resArr.size()]);//这个toArray(T[]) return T[] 是个小细节看看源码理解下
    }


    //生命游戏
    public void gameOfLife(int[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; i++) {


//太麻烦了。。。当个科普了解一下

            }
        }
    }


    public boolean canBeEqual(int[] target, int[] arr) {
        if (arr.length != target.length) return false;
        Arrays.sort(target);
        Arrays.sort(arr);

        for (int i = 0; i < target.length; i++) {
            if (arr[i] != target[i]) return false;
        }
        return true;

    }


    public int heightChecker(int[] heights) {
        int[] temp = new int[heights.length];
        int index = 0;
        for (int i : heights) {
            temp[index++] = i;
        }
        Arrays.sort(temp);
        int count = 0;
        for (int i = 0; i < temp.length; i++) {
            if (temp[i] != heights[i]) {
                count++;
            }
        }

        return count;
    }


    public String removeOuterParentheses(String S) {
        Stack<Character> stack = new Stack<>();
        String str = "";
        for (int i = 0; i < S.length(); i++) {
            if (S.charAt(i) == '(') {
                if (!stack.isEmpty()) str += "(";
                stack.push(S.charAt(i));
            }
            if (S.charAt(i) == ')') {
                stack.pop();
                if (!stack.isEmpty()) str += ")";
            }
        }
        return str;
    }


    //leetcode
    //cdeeelot
    public char firstUniqChar(String s) {
        char[] temp = s.toCharArray();
        Arrays.sort(temp);
        HashMap<Character, Integer> map = new HashMap<>();
        int count = 1;
        char res = ' ';
        int j = 0;
        for (int i = 0; i < temp.length; i++) {
            for (j = i + 1; j < temp.length; j++) {
                if (temp[j] == temp[i]) {
                    count++;
                } else {
                    break;
                }
            }
            if (count == 1) {
                map.put(temp[i], 1);
            } else {
                count = 1;
            }
            i = j - 1;
        }
        for (char ch : s.toCharArray()) {
            if (map.containsKey(ch) && map.get(ch) == 1) {
                res = ch;
                break;
            }
        }
        return res;
    }


    public int[] sumEvenAfterQueries(int[] A, int[][] queries) {
        int sum = 0;
        for (int number : A) {
            if (Math.abs(number) % 2 == 0) {
                sum += number;
            }
        }
        int val = 0, index = 0;
        int count = 0;
        int[] res = new int[A.length];
        int res_index = 0;
        for (int i = 0; i < queries.length; i++) {
            val = queries[i][0];
            index = queries[i][1];
            count = A[index] + val;
            if (Math.abs(A[index]) % 2 != 0 && Math.abs(count) % 2 == 0) {
                sum += count;
            } else if (Math.abs(A[index]) % 2 == 0 && Math.abs(count) % 2 == 0) {
                sum += (count - A[index]);
            } else if (Math.abs(A[index]) % 2 == 0 && Math.abs(count) % 2 != 0) {
                sum -= A[index];
            }
            A[index] = count;
            res[res_index++] = sum;
            count = 0;
        }
        return res;
    }


    //dp  找到第N个丑数，妈的绝了
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int index2 = 0, index3 = 0, index5 = 0;
        for (int i = 0; i < n; i++) {
            dp[i] = Math.min(dp[index2] * 2, Math.min(dp[index3] * 3, dp[index5] * 5));
            if (dp[i] == dp[index2] * 2) index2++;
            if (dp[i] == dp[index2] * 3) index3++;
            if (dp[i] == dp[index2] * 5) index5++;
        }
        return dp[n - 1];
    }


    //00344
    //3 5
    public int specialArray(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        int length = nums.length;
        for (int number = 0; number <= nums[length - 1]; number++) {
            for (int i = 0; i < length; i++) {
                if (nums[i] >= number) {
                    count = length - i;
                    if (count == number) {
                        return number;
                    } else {
                        break;
                    }
                }
            }
            count = 0;
        }
        return -1;
    }


    //就这？
    Queue<TreeNode> queue = new ArrayDeque<>();

    public boolean isEvenOddTree(TreeNode root) {
        boolean flag = true;
        flag = bfs(root, 0);
        return flag;
    }

    public boolean bfs(TreeNode root, int level) {
        queue.offer(root);
        int min = Integer.MIN_VALUE;
        int max = Integer.MAX_VALUE;
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                //奇数层
                if (level % 2 != 0) {
                    if (node.val % 2 != 0) {
                        return false;
                    }
                    if (node.val < max) {
                        max = node.val;
                    } else {
                        return false;
                    }
                    //偶数层
                } else {
                    if (node.val % 2 == 0) {
                        return false;
                    }
                    if (node.val > min) {
                        min = node.val;
                    } else {
                        return false;
                    }

                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }

                size--;
            }

            level++;
            min = Integer.MIN_VALUE;
            max = Integer.MAX_VALUE;

        }
        return true;
    }


    public int minimumOneBitOperations(int n) {


        return 1;
    }


    public int lengthOfLIS(int[] nums) {

        int res = 0;
        int[] dp = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }

        return res;

    }


    public int maxDepth(String s) {
        Stack<Character> stack = new Stack<>();
        StringBuilder stringBuilder = new StringBuilder();
        for (char ch : s.toCharArray()) {
            if (ch == '(' || ch == ')') {
                stringBuilder.append(ch);
            }
        }
        if (stringBuilder.length() == 0) return 0;
        stack.push(stringBuilder.charAt(0));
        int max = 1;
        int size = 0;
        for (int i = 1; i < stringBuilder.length(); i++) {
            if (stringBuilder.charAt(i) == '(') {
                stack.push(stringBuilder.charAt(i));
                size = stack.size();
            } else {
                stack.pop();
                size = stack.size();
            }
            if (size > max) max = size;
        }
        return max;
    }


//52. N皇后 II

    public int totalNQueens(int n) {
        HashSet<Integer> col = new HashSet<>();
        HashSet<Integer> diag1 = new HashSet<>();
        HashSet<Integer> dig2 = new HashSet<>();

        return backtrack(0, n, col, diag1, dig2);


    }

    public int backtrack(int row, int n, HashSet<Integer> col, HashSet<Integer> dig1, HashSet<Integer> dig2) {
        if (n == row) {
            return 1;
        }
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (col.contains(i)) {
                continue;
            }
            if (dig1.contains(row - i)) {
                continue;
            }
            if (dig2.contains(row + i)) {
                continue;
            }
            col.add(i);
            dig1.add(row - i);
            dig2.add(row + i);
            count += backtrack(row + 1, n, col, dig1, dig2);
            System.out.println(count);
            col.remove(i);
            dig1.remove(row - i);
            dig2.remove(row + i);
        }
        return count;
    }


    //单词距离
    public int findClosest(String[] words, String word1, String word2) {
        int distance = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                for (int j = i + 1; j < words.length; j++) {
                    if (words[j].equals(word2)) {
                        if ((j - i) < distance) {
                            distance = j - i;
                            break;
                        }

                    }

                }

            } else if (words[i].equals(word2)) {
                for (int j = i + 1; j < words.length; j++) {
                    if (words[j].equals(word1)) {
                        if ((j - i) < distance) {
                            distance = j - i;
                            break;
                        }

                    }

                }
            }

        }

        return distance;
    }


    public int kthFactor(int n, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if (n % i == 0) list.add(i);
        }
        if (list.size() < k) return -1;
        return list.get(k - 1);


    }


    public int[] singleNumbers(int[] nums) {
        Arrays.sort(nums);
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] == nums[i]) {
                    i = j;
                    break;
                } else {
                    list.add(nums[i]);
                    break;
                }

            }
            if (i == nums.length - 1 && nums[i] != nums[i - 1]) {
                list.add(nums[i]);
            }

        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }


    //1409. 查询带键的排列
    //简单模拟
    public int[] processQueries(int[] queries, int m) {
        int[] res = new int[queries.length];
        int index = 0;

        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <= m; i++) {
            list.add(i);
        }
        for (int i = 0; i < queries.length; i++) {
            int number = list.indexOf(queries[i]);
            res[index++] = number;
            list.remove(number);
            list.add(0, queries[i]);
        }
        return res;
    }


    // 950. 按递增顺序显示卡牌

    public int[] deckRevealedIncreasing(int[] deck) {
        int[] res = new int[deck.length];
        Arrays.sort(deck);
        int index = 0;
        res[0] = deck[0];
        for (int i = 1; i < deck.length; i++) {
            while (res[index] != 0)
                index = (index + 1) % deck.length;
            index = (index + 1) % deck.length;
            while (res[index] != 0)
                index = (index + 1) % deck.length;
            res[index] = deck[i];
        }

        return res;
    }


    public int maxLengthBetweenEqualCharacters(String s) {
        int i;
        int count = -1;
        Map<Character, Integer> map = new HashMap<>();
        for (i = 0; i < s.length(); i++) {
            if (!map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), i);
            } else {
                int number = i - map.get(s.charAt(i)) - 1;
                if (number > count) count = number;
            }

        }

        return count == -1 ? -1 : count;

    }


    //老dp玩家了
    public String findLexSmallestString(String s, int a, int b) {
        Integer[] dp = new Integer[1000];
        dp[0] = Integer.valueOf(s);
        int min = dp[0];
        int i = 1;
        while (true) {
            dp[i] = Math.min(dp[i - 1], Math.min(lun(dp[i - 1].toString(), b), sum(dp[i - 1].toString(), a)));
            if (dp[i] == dp[i - 1]) {
                break;
            }
            i++;
        }

        return dp[i - 1].toString();
    }

    public int lun(String s, int b) {
        List<Character> list = new ArrayList<>();
        int len = s.length();
        for (int i = 0; i < s.length(); i++) {
            if (i != len - b) {
                list.add(s.charAt(i));

            } else {
                for (int j = len - 1; j >= len - b; j--) {
                    list.add(0, s.charAt(j));
                }
                break;
            }

        }
        StringBuilder stringBuilder = new StringBuilder();
        while (!list.isEmpty()) {
            stringBuilder.append(list.get(0));
            list.remove(0);
        }
        return Integer.parseInt(stringBuilder.toString());
    }


    public int sum(String s, int a) {
        int len = s.length();
        Integer[] number = new Integer[len];

        for (int i = 0; i < len; i++) {
            if (i % 2 == 0) {
                number[i] = (int) s.charAt(i) - (int) ('0');
            } else {
                Integer num = (int) (s.charAt(i)) - (int) ('0');
                if (num + a < 9) {
                    num = num + a;
                } else {
                    num = (num + a) % 9 - 1;
                }
                number[i] = num;
            }
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < number.length; i++) {
            stringBuilder.append(number[i]);
        }
        return Integer.parseInt(stringBuilder.toString());
    }


    //双指针法
    //"ababcbacadefegdehijhklij"
    //"caedbdedda"
    public List<Integer> partitionLabels(String S) {
        List<Integer> list = new ArrayList<>();
        int i = 0;
        int j = S.length() - 1;
        int number = -1;
        int first = 0;
        for (i = 0; i < S.length(); i++) {
            char ch = S.charAt(i);
            if (i == number) {
                list.add(number - first + 1);
                number = -1;
                first = i + 1;
                continue;
            }
            for (j = S.length() - 1; j >= 0; j--) {
                if (S.charAt(j) == ch) {
                    if (j > number) {
                        number = j;
                    }
                    break;
                }
            }
            if (i == number) {
                list.add(number - first + 1);
                number = -1;
                first = i + 1;
                continue;
            }

        }
        return list;
    }


    public boolean isPalindrome(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        if (list.size() == 1) return true;
        int i = 0;
        int j = list.size() - 1;
        while (i < j) {
            if (list.get(i).equals(list.get(j))) {
                return false;
            } else {
                i++;
                j--;
            }


        }
        if (i < j) {
            return true;
        }
        return true;
    }

//1038. 把二叉搜索树转换为累加树

    int sum = 0;

    public TreeNode bstToGst(TreeNode root) {
        if (root == null)
            return null;
        TreeNode node = root;
        bstToGstCore(node);
        return root;
    }

    public void bstToGstCore(TreeNode root) {
        if (root.right != null) {
            bstToGstCore(root.right);
        }
        root.val += sum;
        sum = root.val;
        if (root.left != null) {
            bstToGstCore(root.left);
        }
    }


    //1024. 视频拼接
    public int videoStitching(int[][] clips, int T) {
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < clips.length; i++) {
            List<Integer> list1 = new ArrayList<>();
            list1.add(clips[i][0]);
            list1.add(clips[i][1]);
            list.add(list1);
        }
        list.sort((o1, o2) -> {
            if (o1.get(0) == o2.get(0)) {
                return o2.get(1) - o1.get(1);
            }
            return o2.get(0) - o1.get(0);

        });
        list.forEach(integers -> {
            System.out.println(integers);
        });
        int start = list.get(0).get(0);
        int end = list.get(0).get(1);
        int count = 0;
        int video = 0;
        if (start != 0) return -1;
        count++;
        video = end - start;
        for (int i = 1; i < list.size(); i++) {
            if (video == T) return count;
            if (list.get(i).get(0) >= end && end <= list.get(i).get(1)) {
                video += list.get(i).get(1) - end;
                end = list.get(i).get(1);
                count++;
            }

        }
        if (video != T) return -1;
        return count;
    }


    public char slowestKey(int[] releaseTimes, String keysPressed) {
        TreeMap<Integer, Character> map = new TreeMap<>((o1, o2) -> {
            return o2 - o1;
        });
        int number = 0;
        int max = 0;
        for (int i = 0; i < keysPressed.length(); i++) {
            if (releaseTimes[i] - number > max) {
                max = releaseTimes[i] - number;
                map.put(max, keysPressed.charAt(i));
            } else if (releaseTimes[i] - number == max) {
                if (keysPressed.charAt(i) > map.get(max)) {
                    map.put(max, keysPressed.charAt(i));
                }
            }

            number = releaseTimes[i];
        }
        return map.get(map.firstKey());
    }


    public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        int m = l.length;
        List<Boolean> res = new ArrayList<>();


        for (int i = 0; i < m; i++) {
            int length = r[i] - l[i] + 1;
            if (length < 2) {
                res.add(false);
                continue;
            } else {
                List<Integer> temp = new ArrayList<>();
                for (int j = 0; j < length; j++) {
                    temp.add(nums[l[i] + j]);

                }
                res.add(cha(temp));
            }

        }
        return res;
    }

    public boolean cha(List<Integer> temp) {
        temp.sort((o1, o2) -> {
            return o1 - o2;
        });
        int dis = temp.get(1) - temp.get(0);
        for (int i = 2; i < temp.size(); i++) {
            if (temp.get(i) - temp.get(i - 1) != dis) {
                return false;
            }
        }

        return true;
    }


    //dp，用np解决不能p的问题
    public int minimumEffortPath(int[][] heights) {
        int row = heights.length;
        int col = heights[0].length;
        int[][] dp = new int[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                dp[i][j] = Integer.MAX_VALUE;
            }
        }
        dp[0][0] = 0;
        for (int k = 0; k < 100; k++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    int h = heights[i][j];
                    if (i - 1 >= 0)
                        dp[i][j] = Math.min(dp[i][j], Math.max(dp[i - 1][j], (int) Math.abs(h - heights[i - 1][j])));
                    if (i + 1 < row)
                        dp[i][j] = Math.min(dp[i][j], Math.max(dp[i + 1][j], (int) Math.abs(h - heights[i + 1][j])));
                    if (j - 1 >= 0)
                        dp[i][j] = Math.min(dp[i][j], Math.max(dp[i][j - 1], (int) Math.abs(h - heights[i][j - 1])));
                    if (j + 1 < col)
                        dp[i][j] = Math.min(dp[i][j], Math.max(dp[i][j + 1], (int) Math.abs(h - heights[i][j + 1])));
                }
            }
        }

        return dp[row - 1][col - 1];
    }


    public int longestMountain(int[] A) {
//先找山脚 然后山顶，在下山
        int length = 0;
        int max = 0;
        for (int i = 0; i < A.length; i++) {
            if (i + 1 < A.length) {
                if (A[i + 1] > A[i]) {
                    length++;
                    for (int j = i + 1; j < A.length; j++) {
                        if (j + 1 < A.length) {
                            if (A[j + 1] > A[j]) {
                                length++;
                            } else if (A[j + 1] < A[j]) {
                                length++;
                                for (int k = j; k < A.length; k++) {
                                    if (k + 1 < A.length) {
                                        if (A[k + 1] < A[k]) {
                                            length++;
                                        } else {
                                            if (max < length) max = length;
                                            length = 0;
                                            break;
                                        }
                                    }
                                }
                                if (max < length) max = length;
                                length = 0;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return max;
    }


    public int singleNumber(int[] nums) {
        Arrays.sort(nums);
        int res = 0;

        for (int i = 0; i < nums.length; i++) {
            if (i == 0) {
                if (i + 1 < nums.length && nums[i + 1] != nums[i]) {
                    return nums[i];
                }
            }
            if (i == nums.length - 1) {


            }

        }
        return res;
    }


    //1493. 删掉一个元素以后全为 1 的最长子数组
    public int longestSubarray(int[] nums) {
        int suma = 0;
        int sumb = 0;
        int max = 0;
        for (int i : nums) {
            if (i == 1) {
                suma++;
                sumb++;
                max = Math.max(max, suma);
            } else if (i == 0) {
                suma = sumb;
                sumb = 0;
            }
        }
        return max == nums.length ? max - 1 : max;
    }


    //547. 朋友圈
    public void dfs(int[][] M, int[] visited, int i) {
        for (int j = 0; j < M.length; j++) {
            if (M[i][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                dfs(M, visited, j);
            }
        }
    }

    public int findCircleNum(int[][] M) {
        int[] visited = new int[M.length];
        int count = 0;
        for (int i = 0; i < M.length; i++) {
            if (visited[i] == 0) {
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }


    //343. 整数拆分
    //dp[i]表示将i分成至少两个整数的最大积(显然有dp[0]=dp[1]=0)，状态转移方程有：dp[i]=max(j*(i-j),j*dp[i-j]);
    public int integerBreak(int n) {
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i < n + 1; i++) {
            int max = 0;
            for (int j = 1; j < i; j++) {
                max = Math.max(max, Math.max(j * (i - j), j * dp[i - j]));
            }
            dp[i] = max;
        }
        return dp[n];
    }


    //486. 预测赢家
    public boolean PredictTheWinner(int[] nums) {
//偶数个数玩家1总能赢
        if (nums.length % 2 == 0) return true;
//奇数进行dp


        return true;
    }


    //199. 二叉树的右视图
    List<Integer> treelist = new ArrayList<>();
    Queue<TreeNode> treequeue = new LinkedList<>();

    public List<Integer> rightSideView(TreeNode root) {

        if (root == null) return null;
        bfs(root, treelist, 1);
        return treelist;
    }

    public void bfs(TreeNode root, List<Integer> list, int number) {
        treequeue.offer(root);
        while (!treequeue.isEmpty()) {
            int size = treequeue.size();
            for (int i = 0; i < size; i++) {
                TreeNode current = treequeue.poll();
                if (current.left != null) {
                    treequeue.offer(current.left);
                }
                if (current.right != null) {
                    treequeue.offer(current.right);
                }
                if (i == size - 1) {
                    treelist.add(current.val);
                }
            }

        }
    }


    //1310. 子数组异或查询
    public int[] xorQueries(int[] arr, int[][] queries) {
        int[] res = new int[queries.length];
        int index = 0;
        for (int i = 0; i < queries.length; i++) {
            Integer num1 = arr[queries[i][0]];
            Integer num2 = arr[queries[i][1]];
            for (int j = queries[i][0]; j <= queries[i][1]; j++) {
                res[index] ^= arr[j];
            }
            index++;
        }
        return res;
    }

//回溯算法

    List<List<Integer>> lists1 = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        dfs(candidates, target, 0);
        return lists1;
    }

    private void dfs(int[] candidates, int target, int index) {
        if (target == 0) {
            lists1.add(new ArrayList<>(path));
            return;
        }
        for (int i = index; i < candidates.length; i++) {
            if (candidates[i] <= target) {
                if (i > index && candidates[i] == candidates[i - 1]) {
                    continue;
                }
                path.add(candidates[i]);
                dfs(candidates, target - candidates[i], i + 1);
                path.remove(path.size() - 1);
            }
        }
    }


    public boolean closeStrings(String word1, String word2) {
        if (word1.length() != word2.length()) {
            return false;
        }
        // 仅当两个字符串字符类别数目、频次数目相等(用排序字符串比较)时返回true
        Map<Character, Integer> map1 = new TreeMap<>();
        Map<Character, Integer> map2 = new TreeMap<>();
        for (int i = 0; i < word1.length(); i++) {
            map1.put(word1.charAt(i), map1.getOrDefault(word1.charAt(i), 0) + 1);
            map2.put(word2.charAt(i), map2.getOrDefault(word2.charAt(i), 0) + 1);
        }
        if (!map1.keySet().toString().equals(map2.keySet().toString())) {
            return false;
        }
        if (!new TreeSet<>(map1.values()).toString().equals(new TreeSet<>(map2.values()).toString())) {
            return false;
        }
        return true;
    }


    //从左到右找第一个比后面紧挨的数大的数然后删除它
//讨厌。
    public String removeKdigits(String num, int k) {
        if (num.length() == k) return "0";
        Stack<Character> stack = new Stack<>();
        int index = 0;
        stack.push(num.charAt(index++));
        while (k > 0) {
            if (num.charAt(index) > stack.peek()) {
                index++;
                k--;
            } else {
                stack.pop();
                stack.push(num.charAt(index));
                index++;
                k--;
            }
        }
        while (index < num.length()) {
            stack.push(num.charAt(index++));

        }
        StringBuilder stringBuilder = new StringBuilder();
        while (!stack.empty()) {
            stringBuilder.append(stack.pop());
        }
        stringBuilder.reverse();
        Integer i = Integer.valueOf(stringBuilder.toString());

        return i.toString();
    }


    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        for (int i = 0; i < T.length - 1; i++) {
            for (int j = i + 1; j < T.length; j++) {
                if (T[j] > T[i]) {
                    res[i] = j - i;
                    break;
                }
            }
        }
        return res;
    }


    public int[] productExceptSelf(int[] nums) {
        int count = 0;
        int sum = 1;
        for (int i : nums) {
            if (i == 0) {
                count++;
                continue;
            } else {
                sum *= i;
            }
        }
        int[] res = new int[nums.length];
        int index = 0;
        if (count > 1) {
            return res;
        } else if (count == 1) {
            for (int i : nums) {
                if (i == 0) {
                    res[index++] = sum;
                } else {
                    index++;
                    continue;
                }
            }
        } else if (count == 0) {
            for (int i : nums) {
                res[index++] = sum / i;
            }

        }
        return res;
    }


    public String[] permutation(String S) {
        List<String> res = new ArrayList<String>();
        int len = S.length();
        if (len == 0) return new String[0];
        boolean[] used = new boolean[len];
        char[] sChar = S.toCharArray();

        StringBuilder path = new StringBuilder(len);

        // 排序是为了后面的剪枝
        Arrays.sort(sChar);

        dfs(res, sChar, len, path, 0, used);
        return res.toArray(new String[0]);
    }

    /**
     * @param res   结果集
     * @param sChar 输入字符数组
     * @param len   字符数组长度
     * @param path  根结点到任意结点的路径
     * @param depth 当前树的深度
     * @param used  使用标记数组
     */
    private void dfs(List<String> res
            , char[] sChar
            , int len
            , StringBuilder path
            , int depth
            , boolean[] used) {
        // 到达叶子结点
        if (depth == len) {
            res.add(path.toString());
            return;
        }

        for (int i = 0; i < len; i++) {
            if (!used[i]) {

                // 根据已排序字符数组, 剪枝
                if (i > 0 && sChar[i] == sChar[i - 1] && !used[i - 1]) {
                    continue;
                }

                path.append(sChar[i]);
                used[i] = true; // 标记选择
                dfs(res, sChar, len, path, depth + 1, used);
                path.deleteCharAt(depth);
                used[i] = false; // 撤销选择
            }
        }
    }

    //1030. 距离顺序排列矩阵单元格
    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        int[][] res = new int[R * C][2];
        int index = 0;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++) {
                int[] xy = {i, j};
                res[index++] = xy;
            }
        Arrays.sort(res, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                int dis1 = Math.abs(o1[0] - r0) + Math.abs(o1[1] - c0);
                int dis2 = Math.abs(o2[0] - r0) + Math.abs(o2[1] - c0);
                return dis1 - dis2;
            }
        });
        return res;
    }


//二叉树寻路
//public List<Integer> pathInZigZagTree(int label) {
//    ArrayList<Integer> integers = new ArrayList<>();//0.初始化存放结果的变量
//    var a = (int) (Math.log(label) / Math.log(2));//2.计算label所在的层
//    while (label > 1) {//5.循环直到遇到特殊情况1
//        integers.add(label);//3.将label的结果添加到数组中
//        label = (int) (3 * Math.pow(2, --a) - label / 2 - 1);//4.计算下一个label的值
//    }
//    integers.add(1);//6.添加特殊情况 1
//    Collections.reverse(integers); //7.翻转数组
//    return integers;//1.返回结果
//}


    //1415. 长度为 n 的开心字符串中字典序第 k 小的字符串
    public String getHappyString(int n, int k) {
        List<String> list = new ArrayList<>();
        char[] ch = {'a', 'b', 'c'};
        backtrack(n, ch, list, "");
        return k > list.size() ? "" : list.get(k - 1);
    }

    public void backtrack(int n, char[] ch, List<String> list, String str) {
        if (str.length() == n) {
            list.add(str);
            return;
        }
        for (int i = 0; i < ch.length; i++) {
            if (str.length() >= 1 && ch[i] == str.charAt(str.length() - 1)) {
                continue;
            }
            backtrack(n, ch, list, str + ch[i]);//加法一般自动回溯哈哈哈哈。
        }

    }


    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> lists = new ArrayList<>();
        List<Integer> list = new ArrayList<>();

        backtrack(k, n, 1, lists, list);
        return lists;
    }

    public void backtrack(int k, int n, int start, List<List<Integer>> lists, List<Integer> list) {
        if (n < 0) return;
        if (list.size() == k && n == 0) {
            lists.add(new ArrayList<>(list));
            return;
        }


        for (int i = start; i <= 9; i++) {
            list.add(i);
            backtrack(k, n - i, i, lists, list);
            list.remove(list.size() - 1);

        }
    }


    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> lists = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(candidates, target, 0, lists, list);
        return lists;


    }

    public void backtrack(int[] candidates, int n, int start, List<List<Integer>> lists, List<Integer> list) {
        if (n < 0) return;
        if (n == 0) {
            lists.add(new ArrayList<>(list));
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (n - candidates[i] < 0) return;
            list.add(candidates[i]);
            backtrack(candidates, n - candidates[i], i, lists, list);
            list.remove(list.size() - 1);

        }
    }


    int count = 0;

    public int numTilePossibilities(String tiles) {
        boolean[] flag = new boolean[tiles.length()];
        char[] ch = tiles.toCharArray();
        Arrays.sort(ch);
        backtrack(ch, 0, flag);

        return count;
    }

    public void backtrack(char[] ch, int len, boolean[] flag) {
        if (len >= ch.length) return;
        for (int i = 0; i < ch.length; i++) {
            if (i >= 1 && ch[i] == ch[i - 1] && !flag[i - 1]) continue;
            if (!flag[i]) {
                flag[i] = true;
                count++;
                backtrack(ch, len + 1, flag);
                flag[i] = false;
            }
        }

    }


    //    输入：s = "RLRRLLRLRL"
//    输出：4
//    解释：s 可以分割为 "RL", "RRLL", "RL", "RL", 每个子字符串中都包含相同数量的 'L' 和 'R'。
    public int balancedStringSplit(String s) {
        int count = 0;
        int numr = 0;
        int numl = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == 'R') numr++;
            if (ch == 'L') numl++;
            for (int j = i + 1; j < s.length(); j++) {
                if (s.charAt(j) == 'R') numr++;
                if (s.charAt(j) == 'L') numl++;
                if (numl == numr) {
                    count++;
                    i = j;
                    numl = 0;
                    numr = 0;
                    break;
                }
            }
        }
        return count;
    }


    public int[] maxDepthAfterSplit(String seq) {


        return null;
    }


    //89. 格雷编码
    public List<Integer> grayCode(int n) {
        List<Integer> list = new ArrayList<>();
        if (n == 0) {
            list.add(0);
            return list;
        }
        StringBuilder stringBuilder = new StringBuilder();
        char[] temp = {'0', '1'};
        backtrack(list, n, stringBuilder, temp);
        return list;
    }

    public void backtrack(List<Integer> list, int n, StringBuilder stringBuilder, char[] temp) {
        if (stringBuilder.length() == n) {
            Integer number = Integer.valueOf(stringBuilder.toString(), 2);
            list.add(number);
            stringBuilder = null;
            return;
        }

        for (int i = 0; i < 2; i++) {
            stringBuilder.append(temp[i]);
            backtrack(list, n, stringBuilder, temp);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);

        }
    }


    //我TM直接粘贴复制。
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(0), pre;
        dummy.next = head;

        while (head != null && head.next != null) {
            if (head.val <= head.next.val) {
                head = head.next;
                continue;
            }
            pre = dummy;

            while (pre.next.val < head.next.val) pre = pre.next;

            ListNode curr = head.next;
            head.next = curr.next;
            curr.next = pre.next;
            pre.next = curr;
        }
        return dummy.next;
    }


    //卡兰特数
    public int numTrees(int n) {
        // 提示：我们在这里需要用 long 类型防止计算过程中的溢出
        long C = 1;
        for (int i = 0; i < n; ++i) {
            C = C * 2 * (2 * i + 1) / (i + 2);
        }
        return (int) C;
    }


    //454. 四数相加 II
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int count = 0;
        int number = 0;
        int len = A.length;
        HashMap<Integer, Integer> hashMapA_B = new HashMap<>();
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                number = A[i] + B[j];
                hashMapA_B.put(number, hashMapA_B.getOrDefault(number, 0) + 1);
            }
        }
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                number = C[i] + D[j];
                count += hashMapA_B.getOrDefault(-number, 0);
            }
        }
        return count;
    }


    //    输入：s = "aba", t = "baba"
//    输出：6
//    解释：以下为只相差 1 个字符的 s 和 t 串的子字符串对：
//            ("aba", "baba")
//            ("aba", "baba")
//            ("aba", "baba")
//            ("aba", "baba")
//            ("aba", "baba")
//            ("aba", "baba")
//    加粗部分分别表示 s 和 t 串选出来的子字符串。
//1638. 统计只差一个字符的子串数目
    public int countSubstrings(String s, String t) {
        int length_s = s.length();
        int length_t = t.length();
        if (length_s < length_t) {
            HashMap<String, Integer> hashMap = new HashMap<>();
            while (length_s != 0) {
                String str = s.substring(0, length_s);
            }
        } else {

        }
        return 1;

    }


    public int maximumWealth(int[][] accounts) {
        int max = 0;

        for (int i = 0; i < accounts.length; i++) {
            int sum = 0;
            for (int j = 0; j < accounts[i].length; j++) {
                sum += accounts[i][j];
            }
            if (sum > max) max = sum;
        }
        return max;
    }


//
//    输入：arr = [1,4,2,5,3]
//    输出：58
//    解释：所有奇数长度子数组和它们的和为：
//            [1] = 1
//            [4] = 4
//            [2] = 2
//            [5] = 5
//            [3] = 3
//            [1,4,2] = 7
//            [4,2,5] = 11
//            [2,5,3] = 10
//            [1,4,2,5,3] = 15


    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> countMap = new HashMap<Integer, Integer>();
        Map<Integer, Integer> endMap = new HashMap<Integer, Integer>();
        for (int x : nums) {
            int count = countMap.getOrDefault(x, 0) + 1;
            countMap.put(x, count);
        }
        for (int x : nums) {
            int count = countMap.getOrDefault(x, 0);
            if (count > 0) {
                int prevEndCount = endMap.getOrDefault(x - 1, 0);
                if (prevEndCount > 0) {
                    countMap.put(x, count - 1);
                    endMap.put(x - 1, prevEndCount - 1);
                    endMap.put(x, endMap.getOrDefault(x, 0) + 1);
                } else {
                    int count1 = countMap.getOrDefault(x + 1, 0);
                    int count2 = countMap.getOrDefault(x + 2, 0);
                    if (count1 > 0 && count2 > 0) {
                        countMap.put(x, count - 1);
                        countMap.put(x + 1, count1 - 1);
                        countMap.put(x + 2, count2 - 1);
                        endMap.put(x + 2, endMap.getOrDefault(x + 2, 0) + 1);
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
    }


    public int[] decrypt(int[] code, int k) {
        int[] res = new int[code.length];
        int length = code.length;
        int number = k;
        if (k == 0) {
            for (int i = 0; i < res.length; i++) {
                res[i] = 0;
            }

        }
        //向后
        if (k > 0) {
            for (int i = 0; i < res.length; i++) {
                while (k != 0) {
                    res[i] += code[(i + k) % length];
                    k--;
                }
                k = number;
            }

        }
        //向前
        if (k < 0) {
            for (int i = 0; i < res.length; i++) {
                while (k != 0) {
                    res[i] += code[(i + length - Math.abs(k)) % length];
                    k++;
                }
                k = number;
            }

        }
        return res;
    }


    public int[] frequencySort(int[] nums) {
        TreeMap<Integer, Integer> map = new TreeMap<>();

        for (int i : nums) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }


        int[] res = new int[map.size()];
        int i = 0;
        while (!map.isEmpty()) {
            res[i++] = map.get(map.firstKey());
            map.remove(map.firstKey());

        }
        return res;
    }


    public String interpret(String command) {
        StringBuilder stringBuilder = new StringBuilder();
        char[] ch = command.toCharArray();
        for (int i = 0; i < ch.length; i++) {
            if (ch[i] == 'G') {
                stringBuilder.append('G');
                continue;
            }
            if (ch[i] == '(' && i + 1 < ch.length && ch[i + 1] != 'a') {
                stringBuilder.append('o');
                i++;
            } else {
                stringBuilder.append("al");
                i += 3;
            }

        }
        return stringBuilder.toString();
    }


    public int maxOperations(int[] nums, int k) {
        Arrays.sort(nums);
        int right = 0;
        int left = nums.length - 1;
        int count = 0;
        while (right < left) {
            int temp = nums[right] + nums[left];
            if (temp < k) {
                right++;
            } else if (temp > k) {
                left--;
            } else if (temp == k) {
                count++;
                right++;
                left--;
            }
        }
        return count;
    }


    public int concatenatedBinary(int n) {
        int res = 0;
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 1; i <= n; i++) {
            stringBuilder.append(Integer.toBinaryString(i));
        }
        BigInteger temp = new BigInteger(Integer.valueOf(stringBuilder.toString(), 10).toString());
        temp.mod(new BigInteger("1000000007"));
        return temp.intValue();
    }


    public List<String> letterCasePermutation(String S) {
        List<String> res = new ArrayList<>();
        char[] charArray = S.toCharArray();
        dfs(charArray, 0, res);
        return res;
    }

    private void dfs(char[] charArray, int index, List<String> res) {
        if (index == charArray.length) {
            res.add(new String(charArray));
            return;
        }

        dfs(charArray, index + 1, res);
        if (Character.isLetter(charArray[index])) {
            charArray[index] ^= 1 << 5;
            dfs(charArray, index + 1, res);
        }
    }


    public boolean rotateString(String A, String B) {
        if (A.length() == 0 && B.length() == 0) return true;
        if (A.length() == 0 || B.length() == 0) return false;
        String temp = A;
        int i = 1;
        String str1 = A.substring(0, i);
        String str2 = A.substring(i, A.length());
        String str3 = str2 + str1;
        if (str3.equals(B)) return true;
        i++;
        while (!str3.equals(A)) {
            str1 = A.substring(0, i);
            str2 = A.substring(i, A.length());
            str3 = str2 + str1;
            if (str3.equals(B)) return true;
            i++;
        }
        return false;
    }


    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> lists = new ArrayList<>();
        if (root == null) return lists;
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int size = queue.size();
        while (!queue.isEmpty()) {
            while (size != 0) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
                size--;
            }
            lists.add(new ArrayList<>(list));
            list.removeAll(list);
            size = queue.size();
        }
        return lists;
    }


    public String frequencySort(String s) {
        HashMap<Character, Integer> hashMap = new HashMap<>();
        for (char ch : s.toCharArray()) {
            hashMap.put(ch, hashMap.getOrDefault(ch, 0) + 1);
        }
        Set<Map.Entry<Character, Integer>> set = hashMap.entrySet();
        List<Map.Entry<Character, Integer>> list = set.stream().sorted((o1, o2) -> {

            return o2.getValue() - o1.getValue();
        }).collect(Collectors.toList());
        StringBuilder stringBuilder = new StringBuilder();
        while (!list.isEmpty()) {
            int number = list.get(0).getValue();
            char ch = list.get(0).getKey();
            list.remove(0);
            while (number != 0) {
                stringBuilder.append(ch);
                number--;
            }
        }
        return stringBuilder.toString();
    }


    public List<List<String>> groupAnagrams(String[] strs) {
        String[] tempstr = strs.clone();
        List<List<String>> lists = new ArrayList<>();
        List<String> list = new ArrayList<>();
        for (int i = 0; i < strs.length; i++) {
            char[] temp = strs[i].toCharArray();
            Arrays.sort(temp);
            strs[i] = String.valueOf(temp);
        }
        boolean[] flag = new boolean[strs.length];
        for (int i = 0; i < strs.length; i++) {
            String str = strs[i];
            if (flag[i] == false) {
                list.add(tempstr[i]);
                flag[i] = true;
            }
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].equals(str) && flag[j] == false) {
                    list.add(tempstr[j]);
                    flag[j] = true;
                }
            }
            if (list.size() != 0)
                lists.add(new ArrayList<>(list));
            list.removeAll(list);
        }

        return lists;
    }


    public int[] kWeakestRows(int[][] mat, int k) {
        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] res = new int[k];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (mat[i][j] == 1) count++;
            }
            map.put(i, count);
            count = 0;
        }
        Set<Map.Entry<Integer, Integer>> set = map.entrySet();
        List<Map.Entry<Integer, Integer>> list = set.stream().sorted((o1, o2) -> {
            return o1.getValue() - o2.getValue();
        }).collect(Collectors.toList());
        int index = 0;
        while (k != 0) {
            res[index++] = list.get(0).getKey();
            list.remove(0);
            k--;
        }
        return res;
    }


    public int findKthLargest(int[] nums, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i : nums) {
            list.add(i);
        }
        list.sort((o1, o2) -> {
            return o2 - o1;
        });
        list.stream().distinct().collect(Collectors.toList());
        int number = 0;
        while (k != 0) {
            number = list.get(0);
            list.remove(0);

            k--;
        }

        return number;
    }


    //最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        if (text1.length() == 0 || text2.length() == 0) return 0;
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }


    public boolean containsNearbyDuplicate(int[] nums, int k) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (Math.abs(i - j) <= k && nums[i] == nums[j]) {
                    return true;
                }
            }
        }
        return false;
    }


    public int missingNumber(int[] nums) {
        int left = 0, right = nums.length - 1;
        // 下雨等于
        while (left <= right) {
            // 防溢出的写法，位运算提速，这里注意位运算的优先级问题，需要用括号括起来
            int mid = left + ((right - left) >> 1);
            if (nums[mid] == mid) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }


    //118. 杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> lists = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        int[][] temp = new int[numRows][numRows + 1];
        if (numRows == 0) return lists;
        int number = 1;
        temp[0][0] = 1;
        number++;
        list.add(1);
        lists.add(new ArrayList<>(list));
        list.removeAll(list);
        for (int i = 1; i < numRows; i++) {
            for (int j = 0; j < number; j++) {
                if (j == 0) {
                    temp[i][j] = 1;
                    list.add(1);
                }
                if (j != 0) {
                    temp[i][j] = temp[i - 1][j - 1] + temp[i - 1][j];
                    list.add(temp[i - 1][j - 1] + temp[i - 1][j]);
                }
            }
            lists.add(new ArrayList<>(list));
            list.removeAll(list);
            number++;
        }
        return lists;
    }


    public int findLUSlength(String a, String b) {
        if (a.equals(b)) return -1;
        return Math.max(a.length(), b.length());
    }


    //JB傻逼题
    public String toGoatLatin(String S) {
        String[] str = S.split(" ");
        for (int i = 0; i < str.length; i++) {
            int number = i + 1;
            if (str[i].charAt(0) == 'A' || str[i].charAt(0) == 'E'
                    || str[i].charAt(0) == 'I' || str[i].charAt(0) == 'O' || str[i].charAt(0) == 'U' || str[i].charAt(0) == 'a' || str[i].charAt(0) == 'e' || str[i].charAt(0) == 'i' || str[i].charAt(0) == 'o' || str[i].charAt(0) == 'u') {
                str[i] += "ma";
                while (number != 0) {
                    str[i] += "a";
                    number--;
                }
            } else {
                str[i] = str[i].substring(1, str[i].length()) + str[i].charAt(0);
                str[i] += "ma";
                while (number != 0) {
                    str[i] += "a";
                    number--;
                }
            }
        }
        String res = "";
        for (String st : str) {
            res += st;
            res += " ";
        }
        return res.substring(0, res.length() - 1);
    }


    //分治，递归
//分治的精髓永远是递归递归的精髓永远是递归树！！记住了
    public int scoreOfParentheses(String S) {
        //如果s长度小于2，或者s只有两个元素但是不是“（）”就返回0
        if (S.length() < 2 || (S.length() == 2 && (S.charAt(0) != '(' || S.charAt(1) != ')'))) {
            return 0;
        }
        //如果s是“（）”就返回1
        if (S.length() == 2 && S.charAt(0) == '(' && S.charAt(1) == ')') {
            return 1;
        }
        int flag = 0;
        int index = 0;
        //找到与第一个左括号“（”匹配的右括号，下标用index保存，这里没用栈，用flag来模拟出栈入栈
        for (int i = 0; i < S.length(); i++) {
            if (S.charAt(i) == '(') {
                flag++;
            } else {
                flag--;
            }
            if (flag == 0) {
                index = i;
                break;
            }
        }
        if (index == S.length() - 1) {//如果S为“（A）”型
            return 2 * scoreOfParentheses(S.substring(1, index));
        } else {//如果S为“A+B”型
            return scoreOfParentheses(S.substring(0, index + 1)) + scoreOfParentheses(S.substring(index + 1));
        }
    }


    //路漫漫其修远兮。
    public void reorderList(ListNode head) {
        LinkedList<ListNode> queue = new LinkedList<>();
        ListNode Head = head;
        while (Head != null) {
            queue.addLast(Head);
            Head = Head.next;
        }

        while (!queue.isEmpty()) {
            if (Head == null) {
                Head = queue.pollFirst();
            } else {
                Head.next = queue.pollFirst();
                Head = Head.next;
            }
            Head.next = queue.pollLast();
            Head = Head.next;
        }
        if (Head != null) {
            Head.next = null;
        }

    }


    public String intToRoman(int num) {
        StringBuilder stringBuilder = new StringBuilder();
        while (num / 1000 != 0) {
            int number = num / 1000;
            stringBuilder.append("M");
            num = num - 1000;
        }

        while (num / 100 != 0) {
            int number = num / 100;
            if (number == 9) {
                stringBuilder.append("CM");
                num = num % 900;
                break;

            }
            if (number == 4) {
                stringBuilder.append("CD");
                num = num % 400;
                break;
            }

            if (number >= 5) {
                stringBuilder.append("D");
                num = num % 500;
            } else {
                stringBuilder.append("C");
                num = num - 100;
            }
        }


        while (num / 10 != 0) {
            int number = num / 10;
            if (number == 9) {
                stringBuilder.append("XC");
                num = num % 90;
                break;
            }
            if (number == 4) {

                stringBuilder.append("XL");
                num = num % 40;
                break;
            }
            if (number >= 5) {
                stringBuilder.append("L");
                num = num % 50;
            } else {
                stringBuilder.append("X");
                num = num - 10;
            }
        }
        while (num != 0) {
            if (num == 9) {
                stringBuilder.append("IX");
                return stringBuilder.toString();
            }
            if (num == 4) {

                stringBuilder.append("IV");
                return stringBuilder.toString();

            }
            if (num >= 5) {
                stringBuilder.append("V");
                num = num - 5;
            } else {
                stringBuilder.append("I");
                num = num - 1;
            }

        }
        return stringBuilder.toString();
    }


    public int minimumDeleteSum(String s1, String s2) {
        int ans[][] = new int[s1.length() + 1][s2.length() + 1];
        for (int i = 1; i <= s1.length(); i++) {
            ans[i][0] = ans[i - 1][0] + s1.charAt(i - 1);
        }
        for (int j = 1; j <= s2.length(); j++) {
            ans[0][j] = ans[0][j - 1] + s2.charAt(j - 1);
        }
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                ans[i][j] = s1.charAt(i - 1) == s2.charAt(j - 1) ? ans[i - 1][j - 1] : Math.min(ans[i - 1][j] + s1.charAt(i - 1), ans[i][j - 1] + s2.charAt(j - 1));
            }
        }
        return ans[s1.length()][s2.length()];
    }


    public List<Integer> largestValues(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        int size = 0;
        int max = -2147483648;
        List<Integer> list = new ArrayList<>();
        if (root == null) return list;
        queue.offer(root);
        while (!queue.isEmpty()) {
            size = queue.size();
            max = -2147483648;
            while (size != 0) {
                TreeNode temp = queue.poll();
                max = max > temp.val ? max : temp.val;
                if (temp.left != null) queue.offer(temp.left);
                if (temp.right != null) queue.offer(temp.right);
                size--;
            }
            list.add(max);
        }
        return list;
    }


    public boolean noCommonLetters(String s1, String s2) {
        // TODO

        for (char ch : s1.toCharArray()) {
            if (s2.indexOf(ch) != -1) return false;
        }
        return true;
    }

    public int maxProduct(String[] words) {
        int n = words.length;
        int maxProd = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (noCommonLetters(words[i], words[j]))
                    maxProd = Math.max(maxProd, words[i].length() * words[j].length());

        return maxProd;
    }


    //11. 盛最多水的容器
    public int maxArea(int[] height) {
        int start = 0;
        int end = height.length - 1;
        int sum = 0;
        while (start < end) {
            int temp = 0;
            if (height[start] > height[end]) {
                temp = (end - start) * height[end];
                end--;
            } else {
                temp = (end - start) * height[start];
                start++;
            }
            if (temp > sum) {
                sum = temp;
            }
        }
        return sum;
    }


    //969. 煎饼排序
    public List<Integer> pancakeSort(int[] arr) {
        List<Integer> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        for (int i : arr) {
            list.add(i);
        }
        Arrays.sort(arr);
        for (int i = arr.length - 1; i >= 0; i--) {
            if (list.size() == 1) {
                break;
            }
            int index = list.indexOf(arr[i]);
            res.add(index + 1);
            List<Integer> temp = new ArrayList<>(list.subList(0, index + 1));
            Iterator<Integer> it = list.listIterator();
            while (it.hasNext()) {
                Integer a = it.next();
                if (a == arr[i]) {
                    it.remove();
                    break;
                }
                it.remove();
            }
            Collections.reverse(temp);
            list.addAll(0, temp);
            Collections.reverse(list);
            res.add(list.size());
            list.remove(list.size() - 1);
        }
        return res;
    }

//1026. 节点与其祖先之间的最大差值

    public int maxAncestorDiff(TreeNode root) {
        int left = maxAncestorDiff(root.left, root.val, root.val);
        int right = maxAncestorDiff(root.right, root.val, root.val);
        return left > right ? left : right;
    }

    public int maxAncestorDiff(TreeNode root, int max, int min) {
        if (root == null) {
            return 0;
        }
        if (root.val > max) {
            max = root.val;
        } else if (root.val < min) {
            min = root.val;
        }
        if (root.left == null && root.right == null) {
            return max - min;
        }
        int left = maxAncestorDiff(root.left, max, min);
        int right = maxAncestorDiff(root.right, max, min);
        return left > right ? left : right;
    }


    //526. 优美的排列
    int res = 0;

    public int countArrangement(int N) {
        int[] arr = new int[N];
        int index = 0;
        boolean[] flag = new boolean[N];
        for (int i = 1; i <= N; i++) {
            arr[index++] = i;
        }
        List<Integer> list = new ArrayList<>();
        track(arr, flag, list);
        return res;
    }

    public void track(int[] arr, boolean[] flag, List<Integer> temp) {
        if (temp.size() == arr.length && is_my_arr(temp)) {
            res++;
            return;
        }
        for (int i = 0; i < arr.length; i++) {
            if (!flag[i] && is_my_arr(temp)) {
                flag[i] = true;
                temp.add(arr[i]);
                track(arr, flag, temp);
                //这一步进行回溯哦!
                flag[i] = false;
                temp.remove(temp.size() - 1);
            } else {
                continue;
            }
        }
    }

    public boolean is_my_arr(List<Integer> temp) {
        for (int i = 0; i < temp.size(); i++) {
            if (temp.get(i) % (i + 1) == 0 || (i + 1) % temp.get(i) == 0) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }


    public int minSetSize(int[] arr) {
        TreeMap<Integer,Integer> map=new TreeMap<>();
        for (int i = 0; i < arr.length; i++) {
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);

        }
        List<Map.Entry<Integer, Integer>> list = new ArrayList<Map.Entry<Integer, Integer>>(map.entrySet());
        Collections.sort(list,new Comparator<Map.Entry<Integer,Integer>>() {
                //降序排序
                public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                    return o2.getValue().compareTo(o1.getValue());
                }
        });
        Iterator<Integer> it=map.values().iterator();
        int res=0;
        int count=0;
        int size=list.size();
        while(size!=0){
            int number=list.get(0).getValue();
            res+=number;
            count++;
            if(res>=arr.length/2){
                return count;
            }
            list.remove(0);
            size--;
        }
        return 1024;
    }


//剑指 Offer 32 - I. 从上到下打印二叉树
    /**
    public int[] levelOrder(TreeNode root) {
        List<Integer> list=new ArrayList<>();
        Queue<TreeNode> queue=new LinkedList<>();
         if(root==null)  return new int[0];
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode temp=queue.poll();
            list.add(temp.val);
            if(temp.left!=null){
                queue.offer(temp.left);
            }
            if(temp.right!=null){
                queue.offer(temp.right);
            }

        }
int [] res=new int[list.size()];
        int index=0;
        for (int i = 0; i < list.size(); i++) {
            res[index++]=list.get(i);
        }
        return res;
    }
*/

//62. 不同路径
    public int uniquePaths(int m, int n) {
        int [][]dp=new int[m][n];
        dp[0][0]=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m-1][n-1];
    }

//超级丑数


        public int nthSuperUglyNumber(int n, int[] primes) {
            int [] dp=new int[n];
            dp[0]=1;

            int k=primes.length;
            int []index=new int[k];



            for(int i=1;i<n;i++){
                int min=Integer.MAX_VALUE;
                for(int j=0;j<k;j++){
                    if(min>dp[index[j]]*primes[j]){
                        min=dp[index[j]]*primes[j];
                    }
                }
                dp[i]=min;
                //滑动index
                for(int j=0;j<k;j++){
                    if(min==dp[index[j]]*primes[j]){
                        index[j]++;
                    }
                }
            }
            return dp[n-1];
        }



//最小失配数  ps:后面几道题全是经典dp题
//dp[i]表示前i个字符失配次数
//甜姨的代码永远好使（dp小菜鸡滴神
        public int respace(String[] dictionary, String sentence) {
            Set<String> dict = new HashSet<>(Arrays.asList(dictionary));
            int n = sentence.length();
            int[] dp = new int[n + 1];
            for (int i = 1; i <= n; i++) {
                dp[i] = dp[i - 1] + 1;
                for (int idx = 0; idx < i; idx++) {
                    if (dict.contains(sentence.substring(idx, i))) {
                        dp[i] = Math.min(dp[i], dp[idx]);
                    }
                }
            }
            return dp[n];
        }



    public int minimumTotal(List<List<Integer>> triangle) {

        int []dp=new int[triangle.size()+1];
        for(int i=triangle.size()-1;i>=0;i--){
            for(int j=0;j<triangle.get(i).size();j++){
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }



    public int maxSumAfterPartitioning(int[] A, int K) {
        int n = A.length;
        int[] dp = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            int j = i - 1;
            int max = dp[i];
            while ((i - j) <= K && j >= 0) {
                max = Math.max(max, A[j]);
                dp[i] = Math.max(dp[i], dp[j] + (i - j) * max);
                j--;
            }
        }
        return dp[n];
    }

//矩阵类dp类似捡苹果，杨辉三角的题目。
            public int minFallingPathSum(int[][] A) {
                    int [][] dp=new int[A.length][A.length];
                    for(int i=0;i<A.length;i++){
                        for(int j=0;j<A[i].length;j++){
                            if(i==0){dp[i][j]=A[i][j];continue;}
                            if(j==0){dp[i][j]=Math.min(dp[i-1][j],dp[i-1][j+1])+A[i][j];continue;}
                            if(j==A.length-1){dp[i][j]=Math.min(dp[i-1][j],dp[i-1][j-1])+A[i][j];continue;}
                            dp[i][j]=Math.min(Math.min(dp[i-1][j],dp[i-1][j-1]),dp[i-1][j+1])+A[i][j];
                        }
                    }
                    int min=Integer.MAX_VALUE;
                    for(int i=0;i<A.length;i++)
                    {
                        min=Math.min(min,dp[A.length-1][i]);
                    }
            return min;
            }

    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
     int res1=0,res2=0;
     for(int i:distance){
         res1+=i;
     }
int s=Math.min(start,destination);
     int d=Math.max(start,destination);
     for(int i=s;i<d;i++){
         res2+=distance[i];
     }

        return Math.min(res2,res1-res2);
    }


    public String reverseStr(String s, int k) {
    StringBuilder stringBuilder=new StringBuilder();
    StringBuilder res=new StringBuilder();
    if(s.length()<k){
        stringBuilder.append(s).reverse();
        return stringBuilder.toString();
    }
    if(s.length()<2*k&&s.length()>=k){
        stringBuilder.append(s.substring(0,k)).reverse().append(s.substring(k,s.length()));
        return stringBuilder.toString();
    }
    int length=2*k;
    for(int i=0;i<s.length();i=i+length,stringBuilder.delete(0,stringBuilder.length())){
        if((i+length)>s.length()&&(s.substring(i,s.length()).length())>=k){
            stringBuilder.append(s.substring(i,i+k)).reverse().append(s.substring(i+k,s.length()));
            res.append(stringBuilder);
            continue;
        }
        if((i+length)>s.length()&&(s.substring(i,s.length()).length())<k){
            stringBuilder.append(s.substring(i,s.length())).reverse();
            res.append(stringBuilder);
            continue;
        }
        stringBuilder.append(s.substring(i,i+k)).reverse().append(s.substring(k+i,i+length));
        res.append(stringBuilder);
    }
        return res.toString();
    }


//    Integer prev, ans;
//    public int minDiffInBST(TreeNode root) {
//        prev = null;
//        ans = Integer.MAX_VALUE;
//        dfs(root);
//        return ans;
//    }
//
//    public void dfs(TreeNode node) {
//        if (node == null) return;
//        dfs(node.left);
//        if (prev != null)
//            ans = Math.min(ans, node.val - prev);
//        prev = node.val;
//        dfs(node.right);
//    }




        public int maximumProduct(int[] nums) {
            Arrays.sort(nums);
            return Math.max(nums[nums.length-1]*nums[nums.length-2]*nums[nums.length-3],nums[0]*nums[1]*nums[nums.length-1]);
        }


    public int maxCount(int m, int n, int[][] ops) {
        int a,b;
        int mina=Integer.MAX_VALUE,minb=Integer.MAX_VALUE;
        for(int i=0;i<ops.length;i++){
        if(ops[i][0]<mina)mina=ops[i][0];
        if(ops[i][1]<minb)minb=ops[i][1];
     }


return minb*mina;
    }


//
//    int max = 0;
//    public int diameterOfBinaryTree(TreeNode root) {
//        public int diameterOfBinaryTree(TreeNode root) {
//            if (root == null) {
//                return 0;
//            }
//            dfs(root);
//            return max;
//        }
//
//    }
//
//    public int dfs(TreeNode root) {
//        if (root.left == null && root.right == null) {
//            return 0;
//        }
//        int leftSize = root.left == null? 0: dfs(root.left) + 1;
//        int rightSize = root.right == null? 0: dfs(root.right) + 1;
//        max = Math.max(max, leftSize + rightSize);
//        return Math.max(leftSize, rightSize);
//    }


    public int maxRepeating(String sequence, String word) {
    int max=0;
    int count=0;
    String temp=word;
        while(word.length()<=sequence.length()){
            if(sequence.contains(word)){
                count++;
                max++;
            }else {
                count++;
            }
            word+=temp;

    }
        return max;
    }




    public int removeDuplicates(int[] nums) {
        if(nums.length==0)return 0;
        if(nums.length==1) return 1;
        int length=0;
        int index=0;
        nums[index++]=nums[0];
        length++;
        for(int i=1;i<nums.length;i++){
        if(nums[i]!=nums[i-1])
        {
            nums[index++]=nums[i];
            length++;
        }

        }

        return index++;
    }






    public int[] arrayRankTransform(int[] arr) {
    int [] arr2=new int[arr.length];
        for (int i = 0; i <arr.length ; i++) {
    arr2[i]=arr[i];
        }
    Arrays.sort(arr2);
        HashMap<Integer,Integer> map=new HashMap<>();
        int index=1;
        for(int i=0;i<arr2.length;i++){
            if(!map.containsKey(arr2[i])){
                map.put(arr2[i],index++);
            }
        }
    for(int i=0;i<arr.length;i++){
        arr[i]=map.get(arr[i]);

    }
        return arr;
    }

////注意用到中序遍历是升序序列
//int min=Integer.MAX_VALUE;
//    TreeNode pre;
//    public int getMinimumDifference(TreeNode root) {
//        dfs(root);
//
//        return min;
//    }
//
//    public void dfs(TreeNode root){
//if(root==null) return ;
//dfs(root.left);
//if(pre!=null)
//    min=Math.min(min,root.val-pre.val);
//pre=root;
//dfs(root.right);
//    }


//    public List<List<Integer>> levelOrderBottom(TreeNode root) {
//List<List<Integer>> lists=new ArrayList<>();
//List<Integer> list=new ArrayList<>();
//if(root==null) return lists;
//Queue<TreeNode> queue=new LinkedList<>();
//queue.offer(root);
//while(!queue.isEmpty()){
//    int size=queue.size();
//    while(size!=0){
//        TreeNode temp=queue.poll();
//        list.add(temp.val);
//        if(temp.left!=null)queue.offer(temp.left);
//        if(temp.right!=null)queue.offer(temp.right);
//        size--;
//    }
//    lists.add(new ArrayList<>(list));
//    list.removeAll(list);
//}
//        Collections.reverse(lists);
//return lists;
//    }


    public List<Boolean> prefixesDivBy5(int[] A) {

       List<Boolean> res=new ArrayList<>();
       int number=0;
        for(int i=0;i<A.length;i++){
        number=number<<2+A[i];
        number%=10;
            res.add(number%5 == 0);
        }

            return res;
    }

//    public int  find(int root,int [] parent){
//        int res_root=root;
//        while(parent[res_root]!=-1){
//        res_root=parent[res_root];
//        }
//        return res_root;
//    }
//
//    public int union(int x,int y,int [] parent,int [] rank){
//        int x_root=find(x,parent);
//        int y_root=find(y,parent);
//        if(x_root==y_root){
//            return 0;
//        }
//        if(rank[y_root]<rank[x_root]){
//            parent[y_root]=x_root;
//            return 1;
//        }
//        if(rank[y_root]>rank[x_root]){
//            parent[x_root]=y_root;
//            return 1;
//        }
//        if(rank[y_root]==rank[x_root]){
//            parent[y_root]=x_root;
//            rank[x_root]++;
//        }
//        return 1;
//    }
//
//    public int[] findRedundantConnection(int[][] edges) {
//        int [] rank=new int[edges.length+1];
//        int [] res =new int[2];
//        int [] parent=new int[edges.length+1];
//        for(int i=0;i<parent.length;i++)
//        {
//            parent[i]=-1;
//
//        }
//        for(int i=0;i<edges.length;i++){
//            int x=edges[i][0];
//            int y=edges[i][1];
//            int num=union(x,y,parent,rank);
//            if(num==0){
//                res[0]=x;
//                res[1]=y;
//                return res;
//            }
//        }
//        return res;
//    }



        public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
            int n = s.length();
            int[] parent = new int[n];
            // 初始化并查集，一开始任何位置的父母节点都是他自己
            for(int i = 0; i < n; ++i){
                parent[i] = i;
            }
            // 将可以相互交换的位置合并
            for(int i = 0; i < pairs.size(); ++i){
                List<Integer> pair = pairs.get(i);
                union(parent, pair.get(0), pair.get(1));
            }
            // 遍历s字符串，将相同root的字符都扔进一个优先级队列
            Map<Integer, PriorityQueue<Character>> map = new HashMap<>();
            for(int i = 0; i < n; ++i){
                int root = find(parent, i);
                if(!map.containsKey(root)){
                    map.put(root, new PriorityQueue<Character>());
                }
                map.get(root).add(s.charAt(i));
            }
            StringBuilder res = new StringBuilder(n);
            // 构造新字符串
            for(int i = 0; i < n; ++i){
                int root = find(parent, i);
                res.append(map.get(root).poll());
            }
            return res.toString();
        }

        public int find(int[] parent, int a){
            // 递归找到位置节点的root，并赋值，路径压缩
            if(parent[a]!=a){
                parent[a] = find(parent, parent[a]);
            }
            return parent[a];
        }

        public void union(int[] parent, int a, int b){
            int rootA = find(parent, a);
            int rootB = find(parent, b);
            if(rootA != rootB){
                parent[rootA] = rootB;
            }
        }

    public List<String> summaryRanges(int[] nums) {
        List<String> ans = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < nums.length; ++i){
            if(!(i + 1 < nums.length && nums[i] == nums[i + 1] - 1)){
                if(sb.length() > 0) sb.append("->");
                sb.append(nums[i]);
                ans.add(sb.toString());
                sb = new StringBuilder();
            } else{
                if(sb.length() == 0) sb.append(nums[i]);
            }
        }
        return ans;
    }



    public String replaceWords(List<String> dictionary, String sentence) {
        String [] sen=sentence.split(" ");
        HashMap<String,String> map=new HashMap<>();
        for(String str:sen){
            if(!map.containsKey(str)){
                map.put(str,str);
            }
            for(int i=0;i<dictionary.size();i++){
             int length=dictionary.get(i).length();
             if(length>str.length()){
                 continue;
             }else{
                 String temp=str.substring(0,length);
                 if(temp.equals(dictionary.get(i))){
                     if(map.get(str).equals(str)){
                         map.put(str,dictionary.get(i));
                     }else if(map.get(str).length()>dictionary.get(i).length()){
                         map.put(str,dictionary.get(i));
                     }
                 }else {
                     continue;
                 }
             }
            }
        }
        StringBuilder res=new StringBuilder();
            for(String str:sen){
                res.append(map.get(str)).append(" ");
            }
            res.deleteCharAt(res.length()-1);
        return res.toString();
    }




        public void setZeroes(int[][] matrix) {
            int MODIFIED = -1000000;
            int R = matrix.length;
            int C = matrix[0].length;

            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    if (matrix[r][c] == 0) {
                        // We modify the corresponding rows and column elements in place.
                        // Note, we only change the non zeroes to MODIFIED
                        for (int k = 0; k < C; k++) {
                            if (matrix[r][k] != 0) {
                                matrix[r][k] = MODIFIED;
                            }
                        }
                        for (int k = 0; k < R; k++) {
                            if (matrix[k][c] != 0) {
                                matrix[k][c] = MODIFIED;
                            }
                        }
                    }
                }
            }

            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    // Make a second pass and change all MODIFIED elements to 0 """
                    if (matrix[r][c] == MODIFIED) {
                        matrix[r][c] = 0;
                    }
                }
            }
        }


    public String maximumTime(String time) {
        char [] ch=time.toCharArray();
        for(int i=0;i<ch.length;i++){
            if(i==0&&ch[i]=='?'){
                if(ch[i+1]=='?'){
                    ch[i]='2';
                    continue;
                }
                if(ch[i+1]<='3')ch[i]='2';
                if(ch[i+1]>'3')ch[i]='1';
            }
            if(i==1&&ch[i]=='?'){
                if(ch[i-1]=='?'){
                    ch[i]='3';
                    continue;
                }
                if(ch[i-1]=='0'||ch[i-1]=='1')ch[i]='9';
                if(ch[i-1]=='2')ch[i]='3';
            }
            if(i==3&&ch[i]=='?'){
                ch[i]='5';
            }
            if(i==4&&ch[i]=='?'){
                ch[i]='9';
            }
        }

     String res=new String(ch);
        return res;
    }




    public int minCharacters(String a, String b) {
        int maxsum1=0,maxsum2=0,minsum1=0,minsum2=0,e=Integer.MAX_VALUE;
        char maxa='a',maxb='a',mina='z',minb='z';
        if(a.equals(b)) return 0;
        if(a.length()==b.length()){
            e=0;
            for(int i=0;i<a.length();i++){
                if(a.charAt(i)!=b.charAt(i))e+=2;
            }
        }
        for(char i:a.toCharArray()){
            if(i>=maxa)maxa=i;
            if(i<=mina)mina=i;
        }
        for(char i:b.toCharArray()){
            if(i>=maxb)maxb=i;
            if(i<=minb)minb=i;
        }


        for(char i:b.toCharArray()){
            if(i>maxa)continue;
            maxsum1++;
        }
        for(char i:b.toCharArray()){
            if(i<mina)continue;
            minsum1++;
        }
        int res1=maxsum1>=minsum1?minsum1:maxsum1;

        for(char i:a.toCharArray()) {
            if(i>maxb)continue;
          maxsum2++;
        }
        for(char i:a.toCharArray()) {
            if(i<minb)continue;
            minsum2++;
        }
        int res2=maxsum2>=minsum2?minsum2:maxsum2;
        int e2=res1>=res2?res2:res1;

         return e>=e2?e2:e;
    }



public int longtest(String str,int k){

        HashMap<Character,Integer> map=new HashMap<>();
        for(char ch:str.toCharArray()){
            map.put(ch,map.getOrDefault(ch,0)+1);
        }
        for(char c:map.keySet()){
                if(map.get(c)<k){
                    int max=0;
                    for(String s:str.split(String.valueOf(map.get(c)))){
                        max=Math.max(max,longtest(s,k));
                    }
                return  max;
                }
        }
    return str.length();


}




    public List<String> stringMatching(String[] words) {
        List<String> res=new ArrayList<>();
        for(int i=0;i<words.length;i++){
            String temp=words[i];
            for(int k=0;k<words.length;k++){
                if(k!=i&&words[k].contains(temp)) {
                    res.add(temp);
                    break;
                }
            }
        }

return res;
    }


    public double trimMean(int[] arr) {
        Arrays.sort(arr);
        int count =0;
        double sum = 0;
        for(int i = (int)(arr.length * 0.05);i < (int)(arr.length * 0.95);i++){
            count++;
            sum += arr[i];
        }
        return sum / count;
    }






    public int findShortestSubArray(int[] nums) {
HashMap<Integer,Integer> map=new HashMap<>();
        List<Integer> a=new ArrayList<>();
for(int i:nums){
    map.put(i,map.getOrDefault(i,0)+1);
    a.add(i);
}
int max=0;
List<Integer> list=new ArrayList<>();
for(int i:map.keySet()){
    if(map.get(i)>max){
        max=map.get(i);
        if(!list.isEmpty()){
            list.remove(0);
            list.add(0,i);
        }else{
            list.add(0,i);
        }
    }else if(map.get(i)==max)list.add(0,i);
}

int res=Integer.MAX_VALUE;
while(!list.isEmpty()){
    int kk=max;
    int length=0;
    int number=list.get(0);
    for(int i=0;i<a.size();i++){
        if(a.get(i)==number){
                length++;
                kk--;
                for(int j=i+1;j<a.size();j++){
                    if(a.get(j)==number)kk--;
                    length++;
                    if(kk==0){
                        break;
                    }
                }
                if(length<res&&kk==0)res=length;
                break;
        }
    }
    list.remove(0);
}
return res;
    }


    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if(pushed.length==0&&popped.length==0)return true;
        Stack<Integer> stack=new Stack<>();
        stack.push(pushed[0]);
        int index=1;
        for(int i:popped){
            if(stack.empty())stack.push(pushed[index++]);
            while(!stack.empty()&&i!=stack.peek()){
                if(index>=pushed.length)return false;
                stack.push(pushed[index++]);
            }
            stack.pop();
        }
        return true;
    }







    @Test
    public void test() {

        Main main = new Main();
      //  main.minSetSize(new int[]{3,3,3,3,5,5,5,2,2,7});
//        main.arrayRankTransform(new int[]{40,10,20,30});
        //main.stringMatching(new String[]{"mass","as","hero","superhero"});
       // main.findShortestSubArray(new int[]{1,2,2,3,1,4,2});
    }



}
