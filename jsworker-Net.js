/*2020.2.16 lucky_lai 简单神经网络算法*/

/*主函数*/
function main(args){
    try{
        switch(args.action){
            case "init":
                
                var config = {}
                config.input = args.input;
                config.output = args.output;
                config.hidden = args.hidden;
                config.speed = args.speed;
                config.weight1 = initWeight(config.input,config.hidden);
                config.weight2 = initWeight(config.hidden,config.output);
                config.inputValue = initArray(config.input);
                config.hiddenValue = initArray(config.hidden);
                config.outputValue = initArray(config.output);
                config.outputValueExpect = args.outputValueExpect
                config.action = "train";
                return config
            case "train":
                var config = {}
                config.input = args.input;
                config.output = args.output;
                config.hidden = args.hidden;
                config.speed = args.speed;
                config.weight1 = args.weight1;
                config.weight2 = args.weight2;
                config.inputValue = args.inputValue;
                config.hiddenValue = initArray(config.hidden);
                config.outputValue = initArray(config.output);
                config.outputValueExpect = args.outputValueExpect;
                config.action = "train";
                forward(config);
                backward(config);
                return config
        }

    }catch(err){
        console.log(err) 
    }
}






/*random初始化一个inp*oup的二维数组*/
function initWeight(inp,oup){
    var weight = new Array(inp);
    for(var i = 0 ; i < inp ; i++){
        weight[i] = new Array(oup);
        for(var j = 0; j < oup ; j++)weight[i][j]=Math.random();
    }
    return weight;
}
/*初始化一个长度为length的数组，元素全部为0*/
function initArray(length){
    var array = new Array(length);
    for(var i = 0 ; i < length ; i++){
        array[i] = 0;
    }
    return array;
}
/*代价函数*/
function sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}
/*代价函数的导数*/
function sigmoidPrime(x){
    return x * (1 - x);
}
/*计算出output*/
function forward(self){
    self.hiddenValue = initArray(self.hidden);
    self.outputValue = initArray(self.output);
    for(var i = 0 ; i < self.input;i++){
        for(var j = 0 ; j < self.hidden;j++){
            self.hiddenValue[j] += self.inputValue[i]*self.weight1[i][j];
        }
    }
    for(var i = 0 ; i < self.hidden ; i++)self.hiddenValue[i]=sigmoid(self.hiddenValue[i]);
    for(var i = 0 ; i < self.hidden ; i++){
        for(var j = 0;j < self.output;j++){
            self.outputValue[j]+=self.hiddenValue[i]*self.weight2[i][j];
        }
    }
    for(var i = 0; i < self.output;i++)self.outputValue[i]=sigmoid(self.outputValue[i]);
    return;
}
/*反向传播*/
function backward(self){
    var error1 = [];
    var delta1 = [];
    var error2 = [];
    var delta2 = [];
    for(var i = 0;i < self.output; i++){
        error1.push(self.outputValueExpect[i]-self.outputValue[i]);
        delta1.push(error1[i]*sigmoidPrime(self.outputValue[i]));
    }
    var turnedWeight2 = turnMatrix(self.weight2);
    error2 = arrayDot(delta1,turnedWeight2);
    for(var i = 0;i < self.hidden;i++)delta2.push(error2[i]*sigmoidPrime(self.hiddenValue[i]));
    for(var i = 0;i < self.inputValue;i++){
        for(var j = 0; j < self.hidden; j++)self.weight1[i][j]+=self.speed*self.inputValue[i]*delta2[j];
    }
    for(var i = 0; i < self.hidden;i++){
        for(var j = 0; j < self.output;j++)self.weight2[i][j]+=self.speed*self.hiddenValue[i]*delta1[j];
    }


    return;
}
/*矩阵翻转*/
function turnMatrix(array){
    var x = array.length;
    var y = array[0].length;
    var array2 = initWeight(y,x)
    for(var i = 0 ; i < y ; i++){
        for(var j = 0 ; j < x; j++)array2[i][j]=array[j][i]
    }
    return array2;
}
/*列矩阵与矩阵点乘*/
function arrayDot(array1,array2){
    var array = initArray(array2[0].length);
    for(var i = 0 ; i < array2.length;i++){
        for(var j = 0; j < array2[0].length;j++)array[j]+=array1[i]*array2[i][j]
    }
    return array;
}


