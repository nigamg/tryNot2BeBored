package com.facebook;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import javax.print.DocFlavor.CHAR_ARRAY;

import org.quickfixj.java4.edu.emory.mathcs.backport.java.util.Arrays;

public class Solution {
	
    public final int NO_OF_CHARS = 256;
	public static PriorityQueue<Integer> minHeap = new PriorityQueue<>();
	public static PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(new Comparator<Integer>() {

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			return o2.compareTo(o1);
		}
	});
	public static void main(String [] args){
		System.out.println("786... ");
		int [] input = {5, 4, 0, 3, 0, 6, 2};
		System.out.println("... " + Solution.printLongest(input));
		
		int i = 5;
		i++;
		System.out.println("---- "+ (1<<3));
		i++;
		System.out.println("====== "+i);
		
		Solution s = new Solution();
		s.moveZeros(input);
		
		for(int j : input){
			System.out.println(" " + j );
		}
		
	}
	
	public static int printLongest(int [] input){
		if (input == null || input.length == 0)
			return 0;
		
		int [] visited = new int[input.length];
		int maxLen = 0;
		
		for (Integer i: input){
			int currLen = 0;
			while(visited[i] == 0){
				visited[i] = 1;
				currLen = currLen + 1;
				i = input[i];
			}
			maxLen = Math.max(currLen, maxLen);
		}
		
		return maxLen;
	}
	
	public int medianInIntgerStreams(){
		return 0;
	}
	
	public static void reBalance(){
		if(Math.abs(minHeap.size() - maxHeap.size()) > 1){
			PriorityQueue<Integer> biggerHeap = (minHeap.size() > maxHeap.size()) ? minHeap : maxHeap;
			PriorityQueue<Integer> smallerheap =   (minHeap.size() > maxHeap.size()) ? maxHeap : minHeap;
			
			int element = biggerHeap.remove();
			smallerheap.add(element);
		}
	}
	// i<j<k => ai < ak < aj
	public boolean find132Pattern(int [] nums){
		if(nums.length < 3)
			return false;
		int [] min = new int [nums.length];
		Stack<Integer> stack = new Stack<>();
		min[0] = nums[0];
		// compute the min till ith
		for (int i = 1 ; i < nums.length; i ++){
			min[i] = Math.min(min[i], min[i-1]);
		}
		
		for(int j = nums.length-1; j>=0 ; j--){
			if(nums[j] > min[j]){
				while(!stack.isEmpty() && stack.peek() <= min[j]){
					// pop from the stack
					stack.pop();
				}
				if(!stack.isEmpty() && stack.peek() < nums[j]){
					return true;
				}
				
				stack.push(nums[j]);
			}
		}
		return false;
	}
	/***
	 * Find dearrangements
	 * @param n
	 * @return
	 */
	public int findDeArrangement(int n){
		if(n == 0)
			return 1;
		if(n == 1)
			return 0;
		int [] dp = new int[n+1];
		dp[0] = 1;
		dp[1] = 0;
		
		for(int i = 2; i <= n ; i++){
			dp[i] = (int)(((i-1L)*(dp[i-1]+dp[i-2]))%1000000007);
		}
		return 0;
	}
	
	
	/***
	 * Find sum of square
	 */
	public boolean judgeSquare(int c){
		for(long a = 0 ; a*a<c; a++){
			double b = Math.sqrt(c-a*a);
			if(b == (int) b){
				return true;
			}
		}
		return false;
	}
	
	/****
	 * using binary search
	 * @param c
	 * @return
	 */
	public boolean judgeSquareBS(int c){
		for (long a = 0 ; a*a< c; a++){
			int b = c - (int)(a*a);
			if(_binarySearch(0, b, b)){
				return true;
			}
		}
		return false;
	}
	
	public boolean _binarySearch(long s, long e, long n){
		if(s>e)
			return false;
		long mid  = s + (e-s)/2;
		if(mid*mid == n){
			return true;
		}
		if(mid*mid > n){
			return _binarySearch(s, mid-1, n);
		}
		
		return _binarySearch(mid+1, e, n);
	}
	
	
	/****
	 * Smallest range in the sortest array
	 */
	public int [] smallestRange(int [][] nums){
		int minx = 0, miny = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
		
		int [] next = new int [nums.length];
		boolean flag = true;
		
		PriorityQueue<Integer> minQ = new PriorityQueue<Integer>((i, j)->nums[i][next[i]]-nums[j][next[j]]);
		for(int i = 0 ; i < nums.length; i ++){
			minQ.offer(i);
			max = Math.max(max, nums[i][0]);
		}
		
		for(int i = 0 ; i < nums.length && flag; i++){
			for(int j = 0 ; j < nums[i].length&& flag; j++){
				int minI = minQ.poll();
				if(miny-minx > max-nums[minI][next[minI]]){
					minx = nums[minI][next[minI]];
					miny = max;
				}
				next[minI]++;
				if(next[minI] == nums[minI].length){
					flag = false;
					break;
				}
				minQ.offer(minI);
				max = Math.max(max, nums[minI][next[minI]]);
			}
		}
		return new int [] {minx, miny};
	}
	
	public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs){
		Map<List<Integer>, Integer> map = new HashMap<>();
		return shopping(price, special, needs, map);
	}
	
	public int shopping(List<Integer> price, List<List<Integer>> special, List<Integer> needs, Map<List<Integer>, Integer> map ){
		if(map.containsKey(needs)){
			return map.get(needs);
		}
		int j = 0 , res = dot(needs, price);
		
		for(List<Integer> s: special){
			ArrayList<Integer> clone = new ArrayList<Integer>(needs);
			for(j = 0 ; j < needs.size(); j ++){
				int diff = clone.get(j)- s.get(j);
				if(diff < 0)
					break;
				clone.set(j, diff);
			}
			if(j == needs.size()){
				res = Math.min(res, s.get(j) + shopping(price, special, clone, map));
			}
		}
		map.put(needs, res);
		
		return res;
	}
	
	public int dot(List<Integer> a, List<Integer> b){
		int sum = 0;
		for(int i = 0; i < a.size(); i++){
			sum += a.get(i) * b.get(i);
		}
		return sum;
	}
	
	
	/****
	 * Average of TreeNode
	 * @param root
	 */
	public List<Double> averageOfLevels(TreeNode root){
		List<Integer> count = new ArrayList<>();
		List<Double> res = new ArrayList<>();
		
		average(root, 0, res, count);
		for(int i = 0 ; i < res.size(); i ++){
			res.set(i, res.get(i)/count.get(i));
		}
		return res;
	}
	/****
	 * Average of linked list
	 * @param a
	 * @param b
	 * @return
	 */
	public void average(TreeNode t, int i , List<Double> sum, List<Integer> count){
		if(t == null)
			return;
		
		if(i < sum.size()){
			sum.set(i, sum.get(i)+t.data);
			count.set(i, count.get(i)+1);
		}else{
			sum.add(1.0 * t.data);
			count.add(1);
		}
		
		average(t.left, i+1, sum, count);
		average(t.right, i+1, sum, count);
	}
	
	
	public List<Double> averageLevelListIterative(TreeNode root){
		List<Double> res = new ArrayList<>();
		Queue<TreeNode> queue = new LinkedList<>();
		
		queue.add(root);
		
		while(!queue.isEmpty()){
			long sum = 0, count =0;
			Queue<TreeNode> tmp = new LinkedList<>();
			while(!queue.isEmpty()){
				TreeNode n = queue.remove();
				sum+=n.data;
				
				count++;
				
				if(n.left != null){
					tmp.add(n.left);
				}
				
				if(n.right != null){
					tmp.add(n.right);
				}
			}
			queue = tmp;
			res.add(sum*1.0/count);
		}
		return res;
	}
	
	/****
	 * Tree Serialization or Tree to String
	 * @author root
	 */
	public String tree2Str(TreeNode t){
		if(t == null)
			return "";
		
		Stack<TreeNode> stack = new Stack<>();
		stack.push(t);
		
		Set<TreeNode> visited = new HashSet<>();
		StringBuilder s = new StringBuilder();
		while(!stack.isEmpty()){
			t = stack.peek();
			
			if(visited.contains(t)){
				stack.pop();
				s.append(")");
			}else{
				visited.add(t);
				s.append("(" + t.data);
				if(t.left == null && t.right != null){
					s.append("()");
				}
				if(t.right != null){
					stack.push(t.right);
				}
				
				if(t.left != null){
					stack.push(t.left);
				}
			}
		}
		return s.substring(1, s.length()-1);
	}
	
	/****
	 * Merge two trees
	 * @param t1
	 * @param t2
	 * @return
	 */
	public TreeNode mergeTwoTreesRecursive(TreeNode t1, TreeNode t2){
		if(t1 == null)
			return t2;
		if(t2 == null)
			return t1;
		t1.data += t2.data;
		t1.left = mergeTwoTreesRecursive(t1.left, t2.left);
		t1.right = mergeTwoTreesRecursive(t1.right, t2.right);
		return t1;
	}
	
	public TreeNode mergeTwoTrees(TreeNode t1, TreeNode t2){
		if(t1 == null)
			return t2;
		
		Stack<TreeNode[]> stack = new Stack<>();
		stack.push(new TreeNode[]{t1, t2});
		
		while(!stack.isEmpty()){
			TreeNode [] t = stack.pop();
			if(t[0] == null || t[1] == null)
				continue;
			
			t[0].data += t[1].data;
			if(t[0].left == null){
				t[0].left = t[1].left;
			}else{
				stack.push(new TreeNode[]{t[0].left, t[1].left});
			}
			
			if(t[0].right == null){
				t[0].right = t[1].right;
			}else{
				stack.push(new TreeNode[]{t[0].right, t[1].right});
			}
		}
		return t1;
	}
	
	/***
	 * generate palindrome
	 * @param s
	 */
	public void generatePalindromes(String s){
		Set<String> set = new HashSet<>();
		permute(s.toCharArray(), 0, set);
	}
	
	public void swap(char [] s, int i , int j ){
		char tmp = s[i];
		s[i] = s[j];
		s[j] = tmp;
	}
	
	public void permute(char [] s, int l, Set<String> set){
		if(l == s.length){
			if(isPalindrome(s)){
				set.add(new String(s));
			}
		}else{
			for(int i = l ; i < s.length; i++){
				// swap
				swap(s, l , i);
				// permute
				permute(s, i+1, set);
				//swap
				swap(s, l, i);
			}
		}
	}
	
	public boolean isPalindrome(char [] s){
		for(int i = 0 ; i < s.length; i++){
			if(s[i] != s[s.length-1-i]){
				return false;
			}
		}
		return true;
	}
	
	public String shortestPalindrome(String s){
		int n = s.length();
		String rev = new StringBuffer(s).reverse().toString();
		
		int j = 0;
		for(int i = 0 ; i < n; i ++){
			if(s.substring(0, n-i) == rev.substring(i)){
				return rev.substring(0, i) + s;
			}
		}
		return "";
	}
	
	public int maxTrapWater(int [] height){
		int ans = 0, current = 0 ;
		Stack<Integer> stack = new Stack<>();
		while(current < height.length){
			while(!stack.isEmpty() && height[current] > height[stack.peek()]){
				int top = stack.peek();
				
				stack.pop();
				// remove the element from the stack
				if(stack.empty())
					break;
				int distance = current - stack.peek() - 1;
				int boundedHeight = Math.min(height[current], height[stack.peek()]) - height[top];
				
				ans+= distance * boundedHeight;
			}
			stack.push(current++);
		}
		return ans;
	}
	
	/***
	 * Count triangles
	 * @param nums
	 * @return
	 */
	public int countTriangles(int [] nums){
		int count = 0;
		Arrays.sort(nums);
		for(int i = 0 ; i < nums.length-2; i++){
			int k = i+2;
			for (int j = i+1; j< nums.length-1 && nums[i] != 0 ; j++){
				while(k < nums.length && nums[i] + nums[j] > nums[k]){
					k++;
				}
				count+=count+ (k-j+1);
			}
		}
		return count;
	}
	
	public String addBoldTags(String s, String [] dict){
		List<int []> list = new ArrayList<>();
		for(String d: dict){
			for(int i =0 ; i<= s.length()-d.length(); i++){
				if(s.substring(i, i+d.length()).equals(d)){
					list.add(new int []{i, i+d.length()-1});
				}
			}
		}
		if(list.size() == 0){
			return s;
		}
		
		// now sort the indices position
		Collections.sort(list, (a, b)->a[0] == b[0] ? a[1]-b[1] : a[0]-b[0]);
		
		int start, prev = 0, end= 0;
		StringBuilder res = new StringBuilder();
		
		for(int i = 0 ; i < list.size(); i ++){
			res.append(s.substring(prev, list.get(i)[0]));
			start =  i;
			end = list.get(i)[1];
			
			while(i<list.size()-1 && list.get(i+1)[0] <= end+1){
				end = Math.max(end, list.get(i+1)[1]);
				i++;
			}
			res.append("<b>").append(s.substring(list.get(start)[0], end+1)).append("</b>");
			prev = end+1;
		}
		res.append(s.substring(end+1, s.length()));
		return res.toString();
	}
	
	public int longestUniqueSubString(String str){
		int n = str.length();
		int currLen = 1;
		
		int maxLen = 1;
		int prevIndex;
		int i ;
		int [] visited = new int [256];
		for(i = 0 ; i < 256; i++){
			visited[i] = -1;
		}
		
		visited[0] = 1;
		
		for(i = 1; i < n ; i ++){
			prevIndex = visited[str.charAt(i)];
			if(prevIndex == -1  || (i-currLen) > prevIndex)
				currLen++;
			else{
				if(currLen > maxLen){
					maxLen = currLen;
				}
				currLen = i - maxLen;
			}
			visited[str.charAt(i)] = i;
		}
		if(currLen > maxLen)
			maxLen = currLen;
		
		return maxLen;
		
	}
	
	public int longestLHS(int [] nums){
		HashMap<Integer, Integer> map = new HashMap<>();
		int res = 0;
		for(int num: nums){
			map.put(num, map.getOrDefault(num, 0)+1);
			
			if(map.containsKey(num+1)){
				res = Math.max(res, map.get(num) + map.get(num+1));
			}
			
			if(map.containsKey(num-1)){
				res = Math.max(res, map.get(num) + map.get(num-1));
			}
		}
		return res;
	}
	

	
	/****
	 * Short distance in maze
	 * @author root
	 *
	 */
	public int shortestDistance(int [][] maze, int [] start, int [] dest){
		int [][] distance = new int [maze.length][maze[0].length];
		for(int [] row: distance){
			java.util.Arrays.fill(row, Integer.MAX_VALUE);
		}
		distance[start[0]][start[1]]= 0;
		distanceDFS(maze, start, distance);
		return distance[dest[0]][dest[1]] == Integer.MAX_VALUE ? -1 : distance[dest[0]][dest[1]];
	}
	
	public void distanceDFS(int [][] maze, int [] start, int [][] distance){
		int [][] dirs = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
		for(int [] dir : dirs){
			int x = start[0] + dir[0];
			int y = start[1] + dir[1];
			int count =0;
			while(x >= 0 && y>=0 && x<maze.length && y<maze[0].length && maze[x][y] == 0){
				x+=dir[0];
				y+=dir[1];
				count++;
			}
			if(distance[start[0]][start[1]] + count < distance[x - dir[0]][y - dir[1]]){
				distance[x - dir[0]][y - dir[1]] = distance[start[0]][start[1]] + count;
				distanceDFS(maze, new int [] {x-dir[0], y-dir[1]}, distance);
			}
			
		}
	}
	
	public int findInteger(int num){
		return findIntgerUtil(0, 0, num, false);
	}
	
	public int findIntgerUtil(int i , int sum, int num, boolean prev){
		if(sum > num)
			return 0;
		if(1<<i > num){
			return 1;
		}
		
		if(prev){
			return findIntgerUtil(i+1, sum, num, false);
		}
		return findIntgerUtil(i+1, sum, num, false) + findIntgerUtil(i+1, sum + (1<<i), num, true);
	}
	
	public String[] findResturant(String [] list1, String [] list2){
		HashMap<String, Integer> map = new HashMap<>();
		for(int i = 0; i<list1.length; i++){
			map.put(list1[i], i);
		}
		
		List<String> res = new ArrayList<>();
		
		int minSum = Integer.MAX_VALUE, sum;
		
		for(int j =0 ; j < list2.length && j<= minSum; j++){
			if(map.containsKey(list2[j])){
				sum = j + map.get(list2[j]);
				if(sum < minSum){
					res.clear();
					res.add(list2[j]);
					minSum = sum;
				}else if (sum == minSum){
					res.add(list2[j]);
				}
			}
		}
		return res.toArray(new String[res.size()]);
	}
	
	public int arrayNesting(int [] nums){
		int res = 0;
		for(int i = 0 ; i < nums.length; i ++){
			if(nums[i] != Integer.MAX_VALUE){
				int start = nums[i], count = 0;
				while(nums[start] != Integer.MAX_VALUE){
					int tmp = start;
					count ++;
					nums[tmp] = Integer.MAX_VALUE;
				}
				res = Math.max(res, count);
			}
		}
		return res;
	}

	
	public class StringIterator{
		String res;
		int ptr = 0, num = 0;
		char ch = ' ';
		public StringIterator(String s){
			this.res = s;
		}
		public char next(){
			if(!hasNext()){
				return ' ';
			}
			if(num == 0){
				ch = res.charAt(ptr++);
				while(ptr<res.length() && Character.isDigit(res.charAt(ptr))){
					num = num * 10 + res.charAt(ptr++)-'0';
				}
			}
			num--;
			return ch;
		}
		
		public boolean hasNext(){
			return ptr != res.length() || num != 0;
		}
		
	}
	
	public int findPeakElement(int [] nums){
		int l = 0, r = nums.length-1;
		while(l<r){
			int mid = l + (r-l)/2;
			if(nums[mid] > nums[mid+1]){
				r = mid;
			}else {
				l = mid+1;
			}
		}
		return l;
	}
	
	public boolean canPlaceFlower(int [] flowerBed, int n){
		int i = 0, count = 0;
		while(i < flowerBed.length){
			if(flowerBed[i] == 0 && (i == 0 || flowerBed[i-1] == 0) && (i == flowerBed.length-1 || flowerBed[i+1] == 0)){
				count ++;
				flowerBed[i] =1;
			}
			i++;
		}
		return count >= n;
	}
	
	public int maxSubArraySum(int [] a){
		int n = a.length;
		int maxSoFar = 0;
		int maxEndingHere = 0;
		for(int i = 0; i < n ; i ++){
			maxEndingHere = maxEndingHere + a[i];
			if(maxEndingHere < 0)
				maxEndingHere = 0;
			else if (maxSoFar < maxEndingHere)
				maxSoFar = maxEndingHere;
		}
		return maxSoFar;
	}
	
	public int minSubArraySum(int s, int [] a){
		int n = a.length;
		int ans = Integer.MAX_VALUE;
		int left = 0 ;
		int sum = 0;
		for(int i = 0 ; i < n ; i ++){
			sum += a[i];
			while(sum >= s){
				ans = Math.min(ans, i+1-left);
				sum-=a[left++];
			}
		}
		return (ans != Integer.MAX_VALUE) ? ans : 0;
	}
	
	public boolean canPermutePalindrome(String s){
		Set<Character> set = new HashSet<>();
		for(int i = 0; i < s.length(); i ++){
			if(!set.add(s.charAt(i))){
				set.remove(s.charAt(i));
			}
		}
		return set.size() <= 1;
	}
	
	
	public boolean isSubTree(TreeNode s, TreeNode t){
		return traverseTree(s, t);
	}
	
	public boolean equalsTree(TreeNode x, TreeNode y){
		if(x == null && y == null)
			return true;
		
		if (x == null || y == null)
			return false;
		
		return (x.data == y.data) && equalsTree(x.left, y.left) && equalsTree(x.right, y.right);
	}
	
	public boolean traverseTree(TreeNode s, TreeNode t){
		return s != null && (equalsTree(s,t) || equalsTree(s.left, t) || equalsTree(s.right, t));
	}
	
	public int distance(int[] a, int[] b) {
        return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
    }
	
	public int minDistanceForSquarrelJob(int height, int width, 
			int [] tree, int [] squ, int [][] nuts){
		int totDist = 0, d = Integer.MIN_VALUE;
		for(int [] nut : nuts){
			totDist += (distance(nut, tree)*2);
			d = Math.max(d, distance(nut, tree)-distance(nut, squ));
		}
		return totDist-d;
	}
	
	
	public int distributeCandies(int [] candies){
		Arrays.sort(candies);
		int count = 1;
		for(int i = 1; i < candies.length && count < candies.length/2; i ++){
			if(candies[i] > candies[i-1])
				count ++;
		}
		return count;
	}
	
	/***
	 * find the longest string from the dictionary
	 * @author root
	 */
	public boolean isSubSequence(String x, String y){
		int j = 0;
		for(int i = 0 ; i < x.length() && j< y.length(); i++){
			if(x.charAt(j) == y.charAt(i)){
				j++;
			}
		}
		return j == x.length();
	}
	
	public String findLongestWord(String s, List<String> d){
		String maxStr = "";
		for(String str: d){
			if(isSubSequence(str, s)){
				if(str.length() > maxStr.length() || (str.length() == maxStr.length() && str.compareTo(maxStr) < 0))
					maxStr = str;
			}
		}
		return maxStr;
	}
	
	public int minDistance(String s1, String s2){
		int [][] memo = new int[s1.length()+1][s2.length()+1];
		return s1.length() + s2.length() - 2 * lcs(s1, s2, s1.length(), s2.length(), memo);
	}
	
	public int lcs(String s1, String s2, int m, int n, int [][] memo){
		if( m == 0 || n == 0){
			return 0;
		}
		
		if(memo[m][n] > 0)
			return memo[m][n];
		
		if(s1.charAt(m-1) == s2.charAt(n-1))
			memo[m][n] = 1 + lcs(s1, s2, m-1, n-1, memo);
		else
			memo[m][n] = Math.max(lcs(s1, s2, m, n-1, memo), lcs(s1, s2, m-1, n, memo));
		return memo[m][n];
	}
	
	
	/***
	 * Longest common subseq with dynamic programming
	 * @author root
	 */
	public int minDistanceDP(String s1, String s2){
		int [][] dp = new int [s1.length()+1][s2.length()+1];
		for(int i = 0 ; i <= s1.length(); i ++){
			for(int j = 0; j <= s2.length(); j++){
				if (s1.charAt(i) == s2.charAt(j)){
					dp[i][j] = 1 + dp[i-1][j-1];
				}else{
					dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
				}
			}
		}
		return s1.length() + s2.length() - 2*dp[s1.length()][s2.length()];
	}
	
	public int findUnSortedSubArray(int [] nums){
		Stack<Integer> stack = new Stack<Integer>();
		int l = nums.length, r = 0;
		for(int i = 0 ; i < nums.length; i ++){
			while(!stack.isEmpty() && nums[stack.peek()] > nums[i]){
				l = Math.max(l, stack.pop());
			}
			stack.push(i);
		}
		
		stack.clear();
		
		for(int i = nums.length-1;  i >= 0; i --){
			while(!stack.isEmpty() && nums[stack.peek()] < nums[i]){
				r = Math.max(r, stack.pop());
			}
			stack.push(i);
		}
		
		return (r-l) > 0 ? r-l+1 :0;
	}
	
	/****
	 * Kill Process
	 * @author root
	 *
	 */
	public List<Integer> killProcess(List<Integer> pid, List<Integer> ppid, int kill){
		HashMap<Integer, List<Integer>> map = new HashMap<>();
		for(int i = 0 ; i < ppid.size(); i ++){
			if(ppid.get(i) > 0){
				List<Integer> l = map.getOrDefault(ppid.get(i), new ArrayList<>());
				
				l.add(pid.get(i));
				map.put(ppid.get(i), l);
			}
		}
		
		Queue<Integer> queue = new LinkedList<>();
		List<Integer> l = new ArrayList<>();
		
		queue.add(kill);
		
		while(!queue.isEmpty()){
			int r = queue.remove();
			l.add(r);
			if(map.containsKey(r)){
				for(int id:map.get(r)){
					queue.add(id);
				}
			}
		}
		return l;
	}
	
	public String nearestPalindrome(String n){
		long num = Long.parseLong(n);
		for(long i = 1;;i++){
			if(isPalindrome(num-i)){
				return ""+ (num-i);
			}
			if(isPalindrome(num+i)){
				return "" + (num+i);
			}
		}
	}
	
	public boolean isPalindrome(long x){
		long t = x, rev = 0;
		while(t > 0){
			rev = rev * 10 + t%10;
			t = t/10;
		}
		return x == rev;
	}
	
	public int findTreeTilt(TreeNode root){
		int [] sum = new int [1];
		traverse(root, sum);
		return sum[0];
		
	}
	
	public int traverse(TreeNode root, int [] sum){
		if(root == null)
			return 0;
		
		int left = traverse(root.left, sum);
		int right =traverse(root.right, sum);
		sum[0] = sum[0] + Math.abs(left - right);
		return left + right + root.data;
	}
	
	public boolean checkInclusion(String s1, String s2){
		if(s1.length() > s2.length())
			return false;
		int [] s1map = new int [26];
		int [] s2map = new int [26];
		
		for(int i = 0; i< s1.length() ; i ++){
			s1map[s1.charAt(i)-'a']++;
			s2map[s1.charAt(i)-'a']++;
		}
		
		for(int i = 0; i <s2.length()-s1.length(); i ++){
			if(matches(s1map, s2map)){
				return true;
			}
			s2map[s2.charAt(i+s1.length())-'a']++;
			s2map[s2.charAt(i)-'a']--;
		}
		return matches(s1map, s2map);
	}
	
	public int [][] matrixReshape(int [][] nums, int r, int c){
		int [][] res = new int [r][c];
		if(nums .length == 0 || r*c != nums.length * nums[0].length)
			return nums;
		int count = 0;
		Queue<Integer> queue = new LinkedList<>();
		for(int i = 0; i < nums.length; i++){
			for(int j = 0 ; j <  nums[0].length; j++){
				queue.add(nums[i][j]);
			}
		}
		
		for(int i = 0; i < r; i++){
			for(int j = 0; j < c ; j++){
				res[i][j] = queue.remove();
			}
		}
		return res;
	}
	
	public boolean matches (int[] s1map, int [] s2map){
		for(int i = 0 ; i < 26 ;i++){
			if(s1map[i] != s2map[i]){
				return false;
			}
		}
		return true;
	}
	
	public int subArraySum(int [] num, int k ){
		int count = 0, sum = 0;
		HashMap<Integer, Integer> map = new HashMap<>();
		map.put(0, 1);
		for(int i = 0 ; i < num.length; i ++){
			sum += num[i];
			if(map.containsKey(sum-k)){
				count += map.get(sum-k);
			}
			map.put(sum, map.getOrDefault(sum, 0) +1);
		}
		return count;
	}
	
	
	/**
	 * +/- to get a given sum
	 * @param nums
	 * @param s
	 * @return
	 */
	public int findTargetSubWays(int [] nums, int s){
		int count = 0;
		return count;
	}
	
	public void calculate(int [] nums, int i , int sum, int s, int count){
		if(i == nums.length){
			if(sum == s)
				count++; 
		}else{
			calculate(nums, i+1, sum+nums[i], s, count);
			calculate(nums, i+1, sum-nums[i], s, count);
		}
	}
	
	public String reverseWords(String input){
		StringBuilder result = new StringBuilder();
		StringBuilder word = new StringBuilder();
		
		for(int i = 0 ; i < input.length(); i++){
			if(input.charAt(i) != ' '){
				word.append(input.charAt(i));
			}else{
				result.append(word.reverse());
				result.append(" ");
				word.setLength(0);
			}
		}
		result.append(word.reverse());
		return result.toString();
	}
	
	/***
	 * Check perfect number
	 * @param num
	 * @return
	 */
	public boolean checkPerfectNumber(int num){
		if (num <= 0)
			return false;
		
		int sum = 0;
		for(int i = 1; i*i < num; i++){
			sum+=i;
			if(i*i != sum){
				sum+=num/i;
			}
		}
		return (sum-num) == num;
	}
	
	public boolean checkPerfectNum(int num){
		if(num < 0)
			return false;
		int sum = 0;
		for(int i = 1; i < num; i++){
			if(num %i == 0){
				sum+=i;
			}
			if(sum > num )
				return false;
		}
		return (sum == num);
	}
	
	public int[] nextGreaterElement(int [] nums){
		int [] res = new int [nums.length];
		Stack<Integer> stack = new Stack<>();
		for(int i = 2* nums.length-1 ; i >= 0 ; i --){
			while(!stack.isEmpty() && nums[stack.peek()] <= nums[i%nums.length]){
				stack.pop();
			}
			
			res[i%nums.length] = stack.empty() ? -1 : nums[stack.peek()];
			stack.push(i%nums.length);
		}
		return res;
	}
	
	public int candies(int [] ratings){
		int [] candies = new int [ratings.length];
		Arrays.fill(candies, 1);
		
		for(int i = 1; i < ratings.length; i ++){
			if(ratings[i] > ratings[i-1]){
				candies[i] = candies[i-1] + 1;
			}
		}
		
		int sum = candies[ratings.length-1];
		
		for(int i = ratings.length-1 ; i>= 0 ; i--){
			if(ratings[i] > ratings[i+1]){
				candies[i] = Math.max(candies[i], candies[i+1]+1);
			}
			sum+=candies[i];
		}
		return sum;
	}
	
	public boolean isInterleaved(char [] a, char [] b, char [] c){
		int m = a.length;
		int n = b.length;
		
		boolean [][] IL = new boolean [m+1][n+1];
		if(m+n != c.length)
			return false;
		
		for(int i = 0 ; i <= m; i++){
			for(int j = 0 ; j <= n; j++){
				if(i == 0 && j == 0)
					IL[i][j] = true;
				else if (i == 0)
					IL[i][j] = IL[i][j-1] && a[j-1] == c[i+j-1];
				else if(j==0)
					IL[i][j] = IL[i-1][j] && a[i-1] == c[i+j-1];
				else
					IL[i][j] = (IL[i-1][j] && a[i-1] == c[i+j-1]) || ( IL[i][j-1] && a[j-1] == c[i+j-1]);
			}
		}
		return IL[a.length][b.length];
	}
	
	public int climbStairs(int n){
		int [] memo = new int [n+1];
		return climbStairsUtil(0, n, memo); 
	}
	
	public int climbStairsUtil(int i, int j , int [] memo){
		if(i > j)
			return 0;
		if(i == j)
			return 1;
		
		if(memo[i] > 0){
			return memo[i];
		}
		
		memo[i] = climbStairsUtil(i+1, j, memo) + climbStairsUtil(i+2, j, memo);
		return memo[i];
	}
	
	public int maximumGap( int [] nums){
		int maxGap = 0;
		Arrays.sort(nums);
		
		for(int i = 0  ; i < nums.length; i ++){
			maxGap = Math.max(maxGap, nums[i+1]-nums[i]);
		}
		return maxGap;
	}
	
	// {0, 1,1, 0,1, 1}
	public int MaxLengthCountigousArr01(int [] nums){
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, -1);
		int maxLen = 0, count = 0;
		for(int i = 0 ; i < nums.length ; i++){
			count += (nums[i] == 1 ? 1 : -1);
			if(map.containsKey(count)){
				maxLen = Math.max(maxLen, i-map.get(count));
			}else{
				map.put(count, i);
			}
		}
		return maxLen;
	}
	
	public int longestIncreasingSubsequence(int [] arr){
		int n = arr.length;
		int [] lis = new int [n];
		int i , j , max = 0;
		for(i = 0 ; i < n ; i ++)
			lis[i] = 1;
		for(i = 1 ; i < n ; i++){
			for ( j = 0; j <n ; j ++){
				if(arr[i] > arr[j] && lis[i] < lis[j] +1){
					lis[i] = lis[j] + 1;
				}
			}
		}
		
		for(i = 0 ; i < n ; i ++){
			if(max < lis[i])
				max = lis[i];
		}
		return max;
	}
	
	
	public int findSplitPoint(int [] arr){
		int n = arr.length;
		int leftSum = 0;
		for(int i = 0 ; i < n ;i ++){
			leftSum+=arr[i];
		}
		int rightSum = 0;
		for(int i = n-1 ; i>= 0 ; i--){
			rightSum +=arr[i];
			leftSum -= arr[i];;
			if(rightSum == leftSum)
				return i;
		}
		return -1;
	}
	
	public List<Interval> mergeInterval(List<Interval> intervals){
		List<Interval> result = new ArrayList<>();
		if(intervals == null || intervals.size() == 0)
			return result;
		
		Collections.sort(intervals, (a, b) -> a.start != b.start ? a.start-b.start : a.end-b.end);
		Interval pre = intervals.get(0);
		for(int i = 0 ; i <intervals.size(); i ++){
			Interval curr = intervals.get(0);
			if(curr.start > pre.end){
				result.add(pre);
				pre = curr;
			}else{
				Interval merged = new Interval(pre.start, Math.max(pre.end, curr.end));
				pre = merged;
			}
		}
		result.add(pre);
		return result;
	}
	
	public void dfsMatrix(int [][] matrix, int i , int j , int [] max, int len){
		max[0] = Math.max(max[0], len);
		
		int m = matrix.length;
		int n = matrix[0].length;
		
		int [] dx = {-1, 0, 1, 0};
		int [] dy = {0, 1, 0, -1};
		
		for(int k = 0 ; k < 4 ; k++){
			int x = i + dx[k];
			int y = j + dy[k];
			
			if(x >= 0 && x< m && y >= 0 && y < n && matrix[x][y] > matrix[i][j])
				dfsMatrix(matrix, x, y, max, len+1);
		}
	}
	
	public int longestValidParentheses(String s){
		int maxLen = 0;
		Stack<Integer> stack = new Stack<>();
		stack.push(-1);
		
		for(int i = 0 ; i < s.length() ; i ++){
			if(s.charAt(i) == '('){
				stack.push(i);
			}else{
				stack.pop();
				if(stack.isEmpty())
					stack.push(i);
				else
					maxLen = Math.max(maxLen, i-stack.peek());
			}
		}
		return maxLen;
	}
	
	public int largestRectangleInHistogram(int [] height){
		if(height == null || height.length == 0)
			return 0;
		Stack<Integer> stack = new Stack<>();
		int max = 0;
		int i = 0;
		
		while(i < height.length){
			if(stack.isEmpty() || height[i] > height[stack.peek()]){
				stack.push(i++);
			}else{
				int p = stack.pop();
				int h = height[p];
				int w = stack.isEmpty()? i : i-stack.peek()-1;
				max  = Math.max(h*w, max);
			}
		}
		
		while(!stack.isEmpty()){
			int p = stack.pop();
			int h = height[p];
			int w = stack.isEmpty()? i : i-stack.peek()-1;
			max = Math.max(h*w, max);
		}
		return max;
	}
	
	public int dfsInMatrix(int i , int j , int [][] grid){
		if(i == grid.length && j == grid[0].length)
			return grid[i][j];
		if(i < grid.length &&  j < grid[0].length){
			int r1 = grid[i][j] + dfsInMatrix(i, j+1, grid);
			int r2 = grid[i][j] + dfsInMatrix(i+1, j, grid);
			return Math.min(r1, r2);
		}
		
		if(i < grid.length){
			return grid[i][j] + dfsInMatrix(i+1, j, grid);
		}
		
		if(j < grid[0].length){
			return grid[i][j] + dfsInMatrix(i, j+1, grid);
		}
		
		return 0;
	}
	
	public void moveZeros(int [] arr){
		for(int lastNonZeroFoundAt = 0, curr = 0;  curr < arr.length; curr++){
			if(arr[curr] != 0){
				swapInArr(arr, lastNonZeroFoundAt++, curr);
			}
		}
	}
	
	public void swapInArr (int [] arr, int i , int j ){
		int tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
	
	public int maxProfit(int [] prices){
		int maxProfit = 0;
		for(int i = 1; i< prices.length; i ++){
			if(prices[i] > prices[i-1]){
				maxProfit+= prices[i]-prices[i-1];
			}
		}
		return maxProfit;
	}
	
	public int maximumSquare(char[][] matrix){
		int rows = matrix.length;
		int cols = matrix[0].length;
		
		int maxLen = 0;
		int [][] dp = new int [rows+1][cols+1];
		
		for(int i = 1; i < rows; i++){
			for(int j = 1; j < cols; j++){
				if(matrix[i-1][j-1] == '1'){
					dp[i][j] = Math.min(Math.min(dp[i][j-1], dp[i][j-1]), dp[i-1][j-1]) + 1;
					maxLen = Math.max(maxLen, dp[i][j]);
				}
			}
		}
		return maxLen * maxLen;
	}
	
	public int maxAreaTapWater(int [] height){
		int maxArea = 0, l = 0, r = height.length-1;
		while(l < r){
			maxArea = Math.max(maxArea, Math.min(height[l], height[r]) * (r-l));
			if(height[l] < height[r]){
				l++;
			}else{
				r--;
			}
		}
		return maxArea;
	}
	
	/***
	 * Next permutation greater than given number
	 * @param nums
	 */
	public void nextPermutation(int [] nums){
		int i = nums.length - 2;
		while(i >= 0 && nums[i+1] <= nums[i]) // from right
			i--;
		if(i >= 0){
			int j = nums.length-1;
			while(j>=0  && nums[j] <=  nums[i]){
				j--;
			}
			swapInArr(nums, i, j);
		}
		
		reverseArray(nums, i+1, nums.length-1);
	}
	
	public void rotateArray(int [] nums, int k){
		k%=nums.length;
		reverseArray(nums, 0, nums.length-1);
		reverseArray(nums, 0, k-1);
		reverseArray(nums, k, nums.length);
	}
	
	public int wiggleMaxLength(int [] nums){
		if(nums.length < 2)
			return nums.length;
		int down = 1, up =1 ;
		for(int i = 1; i < nums.length; i ++){
			if(nums[i] > nums[i-1])
				up = down+1;
			else if(nums[i] < nums[i-1])
				down = up+1;
		}
		return Math.max(down, up);
	}
	
	public boolean isSymmetric(TreeNode root){
		return isMirror(root, root);
	}
	
	public boolean isMirror(TreeNode t1, TreeNode t2){
		if(t1 == null && t2 == null)
			return true;
		if(t1 == null || t2 == null)
			return false;
		return (t1.data == t2.data) && isMirror(t1.right, t2.left) && isMirror(t1.left, t2.right);
	}
	
	public void reverseArray(int [] num, int start , int end){
		int i = start, j = end;
		while( i < j ){
			swapInArr(num, i++, j--);
		}
	}
	
	public int hammingWeight(int n ){
		int sum = 0;
		while(n != 0){
			sum++;
			n&=(n-1);
		}
		return sum;
	}
	
	public void deleteDuplicates(ListNode head){
		ListNode current =  head;
		while(current != null && current.next != null){
			if(current.next.data == current.data){
				current.next = current.next.next;
			}else{
				current = current.next;
			}
		}
	}
	
	public String longestCommonPrefix(String [] strs){
		if(strs == null || strs.length == 0)
			return "";
		return longestCommonPrefix(strs, 0, strs.length-1);
	}
	
	public String longestCommonPrefix(String [] strs, int l, int r){
		if(l == r)
			return strs[l];
		else{
			int mid = (l+r)/2;
			String lcpLeft = longestCommonPrefix(strs, l, mid);
			String lcpRight = longestCommonPrefix(strs, mid+1, r);
			return commonPrefix(lcpLeft, lcpRight);
		}
			
	}
	
	public ListNode removeNthNodeFromTheEnd(ListNode head, int n){
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		
		ListNode first = dummy;
		ListNode second = dummy;
		
		for(int i = 0 ; i < n ; i++)
			first = first.next;
		while(first != null){
			first = first.next;
			second = second.next;
		}
		
		second.next = second.next.next;
		return dummy.next;
		
	}
	
	public int coinChange(int [] coins, int amount){
		int max = amount+1;
		int [] dp = new int [amount+1];
		Arrays.fill(dp, max);
		dp[0] = 0;
		for(int i = 1 ; i<= amount ; i++){
			for (int j = 0 ; j< coins.length; j++){
				if(coins[i] <= i){
					dp[i] = Math.min(dp[i], dp[i-coins[j]]+1);
				}
			}
		}
		
		return dp[amount] > amount ? -1 : dp[amount];
	}
	
	public int nextedListWeightSum(List<ListNode> list, int depth){
		if(list == null || list.size() == 0)
			return 0;
		int sum = 0;
		for(ListNode ni: list){
			if(ni.isIntger()){
				sum += ni.getIntger();
			}else{
				//sum += nextedListWeightSum(ni.getList(), depth+1);
			}
		}
		return sum;
	}
	
	public int lengthOfLongestSubstring(String s){
		int n = s.length(), ans = 0;
		Map<Character, Interval> map = new HashMap<>();
		for(int j = 0 , i = 0 ; j < n ; j ++){
			if(map.containsKey(s.charAt(j))){
				//i = Math.max(map.get(s.charAt(j)), i);
			}
			ans = Math.max(ans, j-i+1);
			//map.put(s.charAt(j), j+1);
		}
		return ans;
	}
	
	public String commonPrefix(String left, String right){
		int min = Math.min(left.length(), right.length());
		for(int i = 0 ; i < min ; i ++){
			if(left.charAt(i) != right.charAt(i))
				return left.substring(0, i);
		}
		return left.substring(0, min);
	}
	
	public class TrieNode{
		private TrieNode [] links;
		private final int R = 26;
		
		private boolean isEnd;
		
		public TrieNode(){
			links = new TrieNode[26] ;
		}
		
		public boolean containsKey(char ch){
			return links[ch -'a'] != null;
		}
		
		public TrieNode get(char ch){
			return links[ch-'a'];
		}
		
		public void put (char ch, TrieNode node){
			links[ch-'a'] = node;
		}
		
		public void setEnd(){
			isEnd = true;
		}
		
		public boolean getEnd(){
			return isEnd;
		}
	}
	
	public class Trie{
		private TrieNode root;
		public Trie(){
			root = new TrieNode();
		}
		
		public void insertWord(String word){
			TrieNode node = root;
			for(int i = 0 ; i < word.length(); i ++){
				char currentChar = word.charAt(i);
				if(!node.containsKey(currentChar)){
					node.put(currentChar, new TrieNode());
				}
				node = node.get(currentChar);
			}
			node.setEnd();
		}
		
		public TrieNode searchPrefix(String word){
			TrieNode node = root;
			for(int i = 0 ; i < word.length(); i ++){
				char currentLetter = word.charAt(i);
				if(node.containsKey(currentLetter)){
					node = node.get(currentLetter);
				}else{
					return null;
				}
			}
			return node;
		}
		
		public TreeNode invertTree(TreeNode root){
			if (root == null)
				return null;
			TreeNode right = invertTree(root.right);
			TreeNode left = invertTree(root.left);
			
			root.left = right;
			root.right = left;
			return root;
		}
		
		public TreeNode invertTreeIterative(TreeNode root){
			if(root == null)
				return null;
			Queue<TreeNode> queue = new LinkedList<TreeNode>();
			while(!queue.isEmpty()){
				TreeNode current = queue.poll();
				// now swap
				TreeNode tmp = current.left;
				current.left = current.right;
				current.right = tmp;
				
				if(current.left != null)
					queue.add(current.left);
				
				if(current.right != null)
					queue.add(current.right);
			}
			return root;
		}
		
		public ListNode addTwoNumbers(ListNode first, ListNode second){
			ListNode dummy = new ListNode(0);
			ListNode p = first, q = second, curr = dummy;
			
			int carry = 0;
			while(p != null || q != null){
				int x = (p!= null) ? p.data : 0;
				int y = (q!=null) ? q.data : 0;
				int sum = carry + x+ y;
				
				carry = sum/10;
				curr.next = new ListNode(sum%10);
				
				curr = curr.next;
				if(p != null)
					p = p.next;
				if(q != null)
					q = q.next;
			}
			
			if(carry > 0){
				curr.next = new ListNode (carry);
			}
			
			return dummy.next;
		}
		
		public ListNode oddEvenList(ListNode head){
			if(head == null) return null;
			ListNode odd = head, even = head.next, evenHead = head;
			while(even != null && even.next != null){
				odd.next = even.next;
				odd = odd.next;
				
				even.next = odd.next;
				even = even.next;
			}
			odd.next = evenHead;
			return head;
		}
		
		public boolean hasCycleLinkedList(ListNode head){
			if(head == null || head.next == null)
				return false;
			ListNode slow = head;
			ListNode fast = head.next;
			
			while(slow != fast){
				if(fast == null || fast.next == null)
					return false;
				slow = slow.next;
				fast = fast.next.next;
			}
			return true;
		}
		
		public ListNode reverseList(ListNode head){
			ListNode prev =null;
			ListNode current = head;
			while(current != null){
				ListNode next = current.next;
				current.next = prev;
				prev = current;
				current = next;
			}
			return prev;
		}
	}
	
	/****
	 * Check whether it is a sub-set problem
	 * @param set
	 * @param n
	 * @param sum
	 * @return
	 */
	public boolean isSubSetSumProblem(int [] set, int n , int sum){
		if (sum == 0)
			return true;
		
		if(n == 0 && sum != 0)
			return false;
		if(set[n-1] > sum)
			return isSubSetSumProblem(set, n-1, sum);
		
		return isSubSetSumProblem(set, n-1, sum) || isSubSetSumProblem(set, n-1, sum-set[n-1]);
	}
	
	/****
	 * Longest common subsquence
	 * @param x
	 * @param y
	 * @param m
	 * @param n
	 * @return
	 */
	public int longestCommonSubSequence(char [] x, char [] y, int m, int n){
		if (m == 0 || n == 0)
			return 0;
		if(x[m-1] ==  y[n-1])
			return 1+longestCommonSubSequence(x, y, m-1, n-1);
		return Math.max(longestCommonSubSequence(x, y, m, n-1), longestCommonSubSequence(x, y, m-1, n));
	}
	
	/****
	 * Longest increasing sub-sequence
	 * @param arr
	 * @param n
	 * @return
	 */
	public int longestIncreasingSubSequence(int [] arr, int n){
		int [] lis = new int [n];
		int i , j , max = 0;
		for(i = 0 ; i < n; i++)
			lis[i] = 1;
		for(i = 1; i < n ; i++){
			for (j = 0 ; j < i ; j++){
				// compute lis in the bottom-up manner
				if(arr[i] > arr[j] && lis[i] < lis[j]+1){
					lis[i] = lis[j]+1;
				}
			}
		}
	
		for(i =0; i < n ; i++){
			if(max < lis[i]){
				max = lis[i];
			}
		}
		return max;
	}
	
	public int longestSubSequenceWithDifferenceOne(int [] arr, int n){
		int [] lis = new int [n];
		int i , j , max = 0;
		for(i = 0 ; i < n; i++)
			lis[i] = 1;
		for(i = 1; i < n ; i++){
			for (j = 0 ; j < i ; j++){
				// compute lis in the bottom-up manner
				if(arr[i] == arr[j]+1 || arr[i] == arr[j]-1){
					lis[i] = Math.max(lis[i], lis[j]+1);
				}
			}
		}
	
		for(i =0; i < n ; i++){
			if(max < lis[i]){
				max = lis[i];
			}
		}
		return max;
	}
	
	/***
	 * No adjacent
	 * @param arr
	 * @return
	 */
	public int maxSumNoAdjacent (int [] arr){
		int n = arr.length;
		int incl = arr[0];
		int excl = 0;
		int exclNew;
		int i ;
		
		for(i = 1; i < n ; i ++){
			// current max excluding i
			exclNew = (incl > excl) ? incl: excl;
			
			// current max including i
			incl = excl + arr[i];
			excl = exclNew;
		}
		return ((incl > excl) ? incl : excl);
	}
	
	public int maxSumNo3Consec(int [] arr){
		int n = arr.length;
		
		int [] sum = new int[n];
		sum[0] = arr[0];
		sum[1] = arr[0] + arr[1];
		sum[2] = Math.max(sum[1], arr[1]+arr[2]);
		
		for(int i = 3 ; i < n ; i++){
			sum[i] = Math.max(Math.max(sum[i-1], sum[i-2]+arr[i]), arr[i]+arr[i-1]+sum[i-3]);
		}
		return sum[n-1];
	}
	
	public int maxSumPairWithDifferenceLessThanK(int [] arr, int k){
		int n =  arr.length;
		Arrays.sort(arr);
		
		int [] dp = new int[n];
		dp[0] = 0;
		for(int i = 1; i < n ; i++){
			dp[i] = dp[i-1];
			if(arr[i]-arr[i-1] < k){
				if(i>= 2){
					dp[i] = Math.max(dp[i], dp[i-2] + arr[i] + arr[i-1]);
				}else{
					dp[i] = Math.max(dp[i], arr[i]+arr[i-1]);
				}
			}
		}
		return dp[n-1];
	}
	
	public static class ProducerConsumer{
		public static void main(String [] args){
			BlockingQueue<Object> sharedQueue = new LinkedBlockingQueue<>();
			
			Thread prodThread = new Thread(new Solution.Producer(sharedQueue));
			Thread consThread = new Thread(new Solution.Consumer(sharedQueue));
			
			prodThread.start();
			consThread.start();
		}
	}
	
	public static class Producer implements Runnable{
		private final BlockingQueue<Object> sQ;
		public Producer(BlockingQueue<Object> sharedQ){
			this.sQ = sharedQ;
		}
		
		@Override
		public void run() {
			int i = 0;
			// TODO Auto-generated method stub
			System.out.println("Produced " + i);
			try {
				sQ.put(i);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
	}
	public static class Consumer implements Runnable{
		private final BlockingQueue<Object> sQ;
		public Consumer(BlockingQueue<Object> sharedQ){
			this.sQ = sharedQ;
		}
		
		@Override
		public void run() {
			int i = 0;
			// TODO Auto-generated method stub
		
			try {
				System.out.println("Produced " + this.sQ.take());
				
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
	}
	
	
	
	
	public Graph cloneGraph(Graph source){
		if(source == null)
			return null;
		LinkedList<Graph> queue = new LinkedList<Graph>();
		queue.add(source);
		
		HashMap<Graph, Graph> hm = new HashMap<Graph, Graph>();
		hm.put(source, new Graph(source.data));
		
		while(!queue.isEmpty()){
			Graph u = queue.poll();
			Graph cloneNodeU = hm.get(u);
			if(u.neightboours != null){
				Graph [] v = u.neightboours;
				for(Graph graphNode : v){
					Graph cloneNodeG = hm.get(graphNode);
					if(cloneNodeG == null){
						queue.add(graphNode);
						cloneNodeG = new Graph(graphNode.data);
						hm.put(graphNode, cloneNodeG);
					}
					
					cloneNodeU.neightboours[cloneNodeU.neightboours.length] = cloneNodeG;
				}
			}
			
		}
		return hm.get(source);
	}
	
	/***
	 * Print string with condition
	 * @param str
	 * @param ch
	 * @param count
	 * @return
	 */
	public String printString(String str, char ch, int count){
		int occ = 0, i ;
		if (count == 0){
			return str;
		}
		
		for (i = 0 ; i < str.length(); i ++){
			if(str.charAt(i) == ch){
				occ++;
			}
			if(occ == count){
				break;
			}
		}
		return str.substring(i+1);
	}
	
	public int factorial (int n){
		int fact = 1;
		for (int i = 2; i <= n ; i++){
			fact*=n;
		}
		return fact;
	}
	
	/***
	 * returns char  count array
	 * @param str
	 * @return
	 */
	public int [] getCharCount(String str){
		int CHAR_ARRAY = 26;
		int [] count = new int [CHAR_ARRAY];
		for(char c: str.toCharArray()){
			count[c-'a']++;
		}
		return count;
	}
	
	public boolean areKAnagrams(String str1, String str2, int k ){
		int [] count1 = this.getCharCount(str1);
		int [] count2 = this.getCharCount(str2);
		
		int count = 0;
		for(int i = 0 ; i < 26; i ++){
			if(count1[i] > count2[i]){
				count+=Math.abs(count1[i]-count2[i]);
			}
		}
		return (count <= k);
	}
	
	public int countNumberOfSquares(int player, int row, int col, int dirX, int dirY){
		int winR1 = 0, winR2 = 0, winC1= 0, winC2=0;
		int ct = 1; // number of the pieces in a row belonging to the player
		int [][] board = null;
		int r, c;
		r = row+dirX;
		c = col+dirY;
		
		while(r >=0  && r < 13 && c >=0 &&  c< 13 && board[r][c] == player){
			ct++;
			r = row+dirX;
			c = col = dirY;
		}
		winR1 =  r-dirX;
		winC1 = c -dirY;
		
		r = row-dirX;
		c = col-dirY;
		
		while(r>=0 && r< 13 && c >= 0 && c< 13 && board[r][c] == player){
			r = row+dirX;
			c = col+dirY;
			ct++;
		}
		
		winR2 = r+dirX;
		winC2 = r+dirY;
		
		return ct;
	}
	
	private boolean isWinnerInBoard(int row, int col){
		int player = 3;
		if(countNumberOfSquares(player, row, col, 1, 0) >= 5)
			return true;
		
		if(countNumberOfSquares(player, row, col, 0, 1) >= 5)
			return true;
		
		if(countNumberOfSquares(player, row, col, 1, -1) >= 5)
			return true;
		
		if(countNumberOfSquares(player, row, col, 1, 1) >= 5)
			return true;
		
		return false;
	}
	
	// Distinct permutation: !n/!a!b!c
	
	public class Graph{
		int data;
		public Graph(int x){
			this.data =x;
		}
		public Graph  []  neightboours = new Graph [205];
	}
	
	public class Interval{
		int start;
		int end;
		
		public Interval (int start, int end){
			this.start = start;
			this.end = end;
		}
	}
	public class TreeNode{
		int data;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){this.data = x;}
	}
	
	public class ListNode{
		int data;
		ListNode next;
		
		public ListNode (int d){
			this.data =d ;
		}
		List<TreeNode> node = new LinkedList<>();
		public List<TreeNode> getList(){
			return node;
		}
		
		public boolean isIntger(){
			return true;
		}
		
		public int getIntger(){
			return -1;
		}
	}
}

//https://leetcode.com/articles/find-permutation/ TODO
//https://leetcode.com/articles/design-in-memory-file-system/ TODO
