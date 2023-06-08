const express = require("express");
const mysql = require('mysql');
const app = express();
const axios = require("axios");
const cors = require('cors');
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
const multer = require("multer");
const multerS3 = require('multer-s3');
const aws = require('aws-sdk');
const bodyParser = require('body-parser');
const crypto = require('crypto'); // Node.js 내장 모듈이며, 여러 해시 함수를 통한 암호화 기능을 제공
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
const fs = require('fs');
const OpenAI = require('openai');
app.use(cors());


// aws 연동
const s3 = new aws.S3({
  region: 'ap-northeast-2',
  accessKeyId: "--",
  secretAccessKey: "--"
});

// sql 연동
const db = mysql.createConnection({
  user: '--',
  host: '--',
  password: '--',
  database: '--'
});

// db가잘 연동 되었는지 확인
db.connect(function(err) { 
  if(err) throw err;
  console.log('DB is Connected!')
});


app.listen(3001, function() {
  console.log("Server is running");
});


//S3연동
const storage = multerS3({
  s3: s3,
  bucket: 'sodam-s', // 자신의 s3 버킷 이름
  contentType: multerS3.AUTO_CONTENT_TYPE,
  acl: 'public-read', // 읽기만 가능, 쓰기 불가능
  metadata: function(req, file, cb) {
      cb(null, {fieldName: file.fieldname});
  },
  key: function (req, file, cb) { // 객체의 키로 고유한 식별자 이기 때문에 겹치면 안됨
      cb(null, `contents/${Date.now()}_${file.originalname}`);
  }
})


const upload = multer({
  storage: storage // storage를 multerS3 객체로 지정
})




// s3에 업로드
app.post('/upload', upload.single('image'), (req, res) => {
  console.log("일단 옴");
  res.send(req.file.location);
  console.log(req.file.location);
})



//이메일 인증번호
app.post('/checkNum', (req, res) => {
  console.log("오긴 옴")
  let id = req.query.id;
  let email = req.query.email;
  const randomBytes = require("crypto").randomBytes(3);
  const number = parseInt(randomBytes.toString("hex"), 16); // 인증번호 생성

  let values = [id, email]
  const sql = "Select email From userInfo Where id = ? AND email = ?";

  db.query(sql, values,(err, result) => {
    if(err){
      console.log(err);
    }
    if(result.length > 0) {
      console.log("계정 존재");
      console.log(number);
      axios({
        method: "POST",
        url: "https://api.emailjs.com/api/v1.0/email/send",
        headers: {
          "Content-Type": "application/json",
        },
        data: {
          service_id: "--",
          template_id: "--",
          user_id: "--",
          accessToken: "--",
          template_params: {
            email: email,
            checkNum: number,
          },
        },
      })
      .then((result) => {
        console.log(number);
        res.send({result:number});
      })
      .catch(function(error) {
        console.log(error)
      })

    }else{
      console.log("계정이 존재하지 않음")
      res.send('1');
    }
    });
});


//인증 성공 시 비밀번호 변경
app.post('/Reset', (req, res) => {
  let id = req.query.id;
  let pw = crypto.createHash("sha512").update(req.query.pw).digest("base64");
  let values = [pw, id]
  
  console.log(values);

  const sql = "UPDATE userInfo SET pw = ? Where id = ?";
  db.query(sql, values,
    (err, result) => {
     
     console.log(result)
        if (err)
            console.log(err);
        else{
          console.log(result);
          res.send(result);
        }
        
    });

});

//이메일 변경
app.post('/changeEmail', (req, res) => {
  let id = req.query.id;
  let email = req.query.email;
  let values = [email, id]
  
  console.log(values);

  const sql = "UPDATE userInfo SET email = ? Where id = ?";
  db.query(sql, values,
    (err, result) => {
     
     console.log(result)
        if (err)
            console.log(err);
        else{
          console.log(result);
          res.send(result);
        }
        
    });

});

//비밀번호 맞는지 확인
app.post('/pwCheck', (req, res) => {
  console.log("비밀번호 확인하러 옴")
  let id = req.query.id;
  let pw = crypto.createHash("sha512").update(req.query.pw).digest("base64");
  let values = [id, pw];

  console.log(values);

  const sql = "Select email From userInfo Where id = ? AND pw = ?";
  db.query(sql, values,
  (err, result) => {
    if(err){
      console.log(err);
    }
    if(result.length > 0){
      console.log(result);
      res.send("0");
    }else {
      console.log(result);
      res.send("1");
    }
    

});
});


//회원가입
app.post('/register', (req, res) => {
  //파라미터를 받아오는 부분
  let id = req.query.id;
  let email = req.query.email;
  let pw = crypto.createHash("sha512").update(req.query.pw).digest("base64");
  let values = [id, email, pw]

  console.log(values)
  
  //SQL 코드
  const sql = "INSERT INTO userInfo(id, email, pw) VALUES(?, ?, ?)"
  db.query(sql, values,
      (err, result) => {
          if (err)
              console.log(err);
          else
              res.send(result);
      });
});

//로그인
app.post('/login', (req, res) => {
  let id = req.query.id;
  let pw = crypto.createHash("sha512").update(req.query.pw).digest("base64");

  let values = [id, pw]

  console.log(values);

  const sql = "select id From userInfo Where id = ? AND pw = ?"
  db.query(sql, values,
    
    (err, result) => {
      
        if (err)
            console.log(err);
        if(result.length > 0) {
          console.log("로그인 성공");
          res.send('0'); // 로그인에 성공하면 0을, 실패하면 1을 반환
          console.log(result) 
        }else {
          res.send('1');
          console.log("로그인 실패");
        }
    });
});


//아이디 중복 체크
app.post('/overlap', function(req, res) {
  let id = req.query.id;
  let values = [id]

  // console.log(values)
  //SQL 코드
  const sql = "Select * From userInfo WHERE id = ?"
  db.query(sql, values,
    
      (err, result) => {
          if (err)
              console.log(err);
          if(result.length < 1) {
            console.log("없는 아이디");
            res.send('0'); // 중복검사에 성공하면 0을, 실패하면 1을 반환
          }else {
            res.send('1');
            console.log("있는 아이디");
            console.log(result)
          }
      });
});

// 플라스크 로컬 통신
app.post('/flask', (req, res) => {
  let text2 = req.query.text;
  let text3 = "너의 이름은 이제부터 소담이야. 귀여운 말투로 말을 해줘. " + text2
  console.log("대화: ", text3)
  axios({
    method: "POST",
    url: "--/getSentiment",   // loacal 로 진행함.
    data: {
      input: text2
    },
  })
  .then(async(result) => {
    // console.log(result.data.sentence);
    // console.log(result.data.emotion);
    // console.log(result.data.index);
    console.log(result.data.max_cos)
    if(result.data.max_cos < 1) {
      const { Configuration, OpenAIApi } = require("openai");

        const configuration = new Configuration({
          apiKey: "--",
        });
        const openai = new OpenAIApi(configuration);

        const completion = await openai.createCompletion({
          model: "text-davinci-003",
          prompt: text3,
          max_tokens: 128,
          temperature: 0.7,
        });

        const result2 = completion.data.choices[0].text;
        console.log("gpt: ",result2);

        const tokenCount = completion.data.usage.total_tokens;
        console.log("Total tokens used:", tokenCount);
        const response = {
          result: result2,
          tokenCount: tokenCount,
        };
        let splitText = response.result.split(/[.,!~?]/);
        let splitText3 = splitText.slice(1, splitText.length - 2).join(".").trim().replace(/\n/g, '') + ".";
        console.log("결과확인: ",splitText[1]);
        let gptText = {"sentence": splitText3, "emotion": result.data.emotion};
        console.log("최종: ",gptText);
        res.send(JSON.stringify(gptText));
    }else {
      res.send(result.data);
    }
  })
  .catch((error) => {
    console.error(error);
    res.status(500).send(error);
  });
});



//글 작성
app.post('/write', async function(req, res2) {
  console.log("글 작성 하러 옴");
  let id = req.query.id;
  let title = req.query.title;
  let content = req.query.content;
  let year = req.query.year;
  let month = req.query.month;
  let day = req.query.day;
  let img = req.query.img;
  let cb_sentence = ""; //최종 gpt결과값 저장
  let cb_emotion = req.query.cb_emotion;
  

  // 감정 키워드 딕셔너리로 각각 몇 개인지 빼옴
  let chart_emotion = cb_emotion.split("##");
  chart_emotion.shift();
  console.log("chart_emotion: ",chart_emotion);
  const count = {};

  for (let i = 0; i < chart_emotion.length; i++) {
    const emotion = chart_emotion[i];
    count[emotion] = (count[emotion] || 0) + 1;
  }

  console.log(count);

//top3 분류해서 각각에 값 적용
const sortedEntries = Object.entries(count).sort((a, b) => b[1] - a[1]);

let top_emotion = null;
let second_emotion = null;
let third_emotion = null;
let top_number = 0;
let second_number = 0;
let third_number = 0

if (sortedEntries.length > 0) {
  top_emotion = sortedEntries[0][0];
  top_number = sortedEntries[0][1];
}

if (sortedEntries.length > 1) {
  second_emotion = sortedEntries[1][0];
  second_number = sortedEntries[1][1];
}

if (sortedEntries.length > 2) {
  third_emotion = sortedEntries[2][0];
  third_number = sortedEntries[2][1];
}

console.log('top_emotion:', top_emotion);
console.log('second_emotion:', second_emotion);
console.log('third_emotion:', third_emotion);


const { Configuration, OpenAIApi } = require("openai");
console.log("message: ", content);
let message = "한국어로 공감 또는 위로를 해줘. " + content

const configuration = new Configuration({
  apiKey: "--",
});
const openai = new OpenAIApi(configuration);

const completion = await openai.createCompletion({
  model: "text-davinci-003",
  prompt: message,
  max_tokens: 1024,
  temperature: 0.7,
});

const result = completion.data.choices[0].text;
console.log("gpt의 결과: ",result);

const tokenCount = completion.data.usage.total_tokens;
console.log("Total tokens used:", tokenCount);
const response = {
  result: result,
  tokenCount: tokenCount,
};

let gptText = response.result.split(".");
let gptText2 = gptText.slice(0, gptText.length).join(".").trim();

cb_sentence = gptText2

console.log("마지막 gpt: ", cb_sentence);


  setTimeout(() => {
    console.log("sql실행")
    let values = [id, title, content, year, month, day, img, cb_sentence, cb_emotion, top_emotion, second_emotion, third_emotion, top_number, second_number, third_number];
    const sql = "INSERT INTO diary(id, title, content, year, month, day, img, cb_sentence, cb_emotion, top_emotion, second_emotion, third_emotion, top_number, second_number, third_number) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    const sql2 = "Select MAX(diarykey) as diarykey From diary";

    db.query(sql, values, (err, result) => {
      if (err) {
        console.log(err);
      } else {
        db.query(sql2, (err, res) => {
          if(err){
            console.log(err);
          }else {
            console.log("다이어리키 가져오나?",res);
            res2.send(res);
          }
        })
      }
    });
  }, 300)

});



//사용자 일기 내용 반환 (일반 목록 리스트)
app.post('/myDiary',(req, res) => {
  console.log("일로 옴");
  let id = req.query.id;
  let year = req.query.year;

  let values = [id, year];
  console.log(values);
  
  const sql= "Select diarykey, title, content, year, month, day, img, cb_sentence, cb_emotion, top_emotion, second_emotion, third_emotion, top_number, second_number, third_number From diary Where id = ? AND year =? Order By day DESC";

  db.query(sql, values,
    (err, result) => {
    //  console.log(result)
        if (err)
            console.log(err);
        else
        // console.log(result);
        res.send(result);
    });
});


//사용자 일기 수정
app.post('/diaryModify', async function(req, res) {
  console.log("수정하러 옴");
  let diarykey = req.query.diarykey;
  let title = req.query.title;
  let content = req.query.content;
  let year = req.query.year;
  let month = req.query.month;
  let day = req.query.day;
  let img = req.query.img;
  let cb_sentence = ""; //최종 gpt결과값 저장
  let cb_emotion = req.query.cb_emotion;


  const { Configuration, OpenAIApi } = require("openai");
console.log("message: ", content);
let message = "한국어로 공감 또는 위로를 해줘. " + content

const configuration = new Configuration({
  apiKey: "--",
});
const openai = new OpenAIApi(configuration);

const completion = await openai.createCompletion({
  model: "text-davinci-003",
  prompt: message,
  max_tokens: 1024,
  temperature: 0.7,
});

const result = completion.data.choices[0].text;
console.log("gpt의 결과: ",result);

const tokenCount = completion.data.usage.total_tokens;
console.log("Total tokens used:", tokenCount);
const response = {
  result: result,
  tokenCount: tokenCount,
};

let gptText = response.result.split(".");
let gptText2 = gptText.slice(0, gptText.length).join(".").trim();

cb_sentence = gptText2

console.log("마지막 gpt: ", cb_sentence);

  let values = [title, content, year, month, day, img, cb_sentence,  diarykey];
  let valuse2 = [cb_emotion, diarykey];
  console.log(values);
  const sql = "Update diary Set title = ?, content = ?, year = ?, month = ? , day = ?, img = ?, cb_sentence =? Where diarykey = ?"
  const sql2 = "UPDATE diary SET cb_emotion = CONCAT(?, cb_emotion) WHERE diarykey = ?"
  db.query(sql, values,
    (err, result) => {
        if (err)
            console.log(err);
        else{
          db.query(sql2, valuse2, (err, result) => {
            if(err) {
              console.log(err)
            }else {
              res.send("3");
            }
          })
        }
    });
})


//사용자 일기 삭제
app.post('/diaryDelete', (req,res) => {
  console.log("삭제하러옴")
  let diarykey = req.query.diarykey;
  let imgKey = req.query.imgKey;

  console.log("imgKey", imgKey)

  let values = [diarykey];
  
  //s3에 저장된 이미지 삭제
  s3.deleteObject({
    Bucket: 'sodam-s3',
    Key: imgKey,
  }, (err, data) => {
    if(err) {
      console.log(err);
    }else {
      console.log(data);
    }
  }
  );

  const sql = "Delete From diary Where diarykey = ?"
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }else {
      res.send(result);
    }
  });
});


//사용자 일기내용 반환(상세페이지)
app.post('/diaryInfo', (req, res) => {
  let diarykey = req.query.diarykey;

  console.log("방금 받았어요: " , diarykey);

  let values = [diarykey];
  let sql = "Select * From diary Where diarykey =?"
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }
  else {
    console.log(result);
      res.send(result);
    }
})
});


//앨범
app.post('/album', (req, res) => {
  console.log("앨범");
  let id = req.query.id;
  let values = [id];

  let sql = "select diarykey, img, title, year, month, day From diary Where id =?"
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }else {
      console.log(result);
      res.send(result);
    }
  })

});

//앨범 개수
app.post('/albumCnt', (req, res) => {
  let id = req.query.id;

  let sql = "select count(img) as cnt from diary where id = ? AND img is not null"
  db.query(sql, id, (err, result) => {
    if(err){
      console.log(err);
    }else {
      console.log(result);
      res.send(result);
    }
  })
});


//일기 초기화
app.post('/deleteAll', (req, res) => {
  console.log("일기 초기화하러 옴");
  let id = req.query.id;
  let values = [id];

  let sql = "delete from diary where id =?";
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }else {
      res.send(result);
    }
  })
});

//계정 탈퇴
app.post('/withdrawal', (req, res) => {
  console.log("계정 탈퇴하러 옴");
  let id = req.query.id;
  let values = [id];

  let sql = "delete From diary where id =?"; 
  let sql2 = "delete From userInfo Where id =?";

  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }else {
      db.query(sql2, values, (err, result) => {
              if(err) {
                console.log(err);
              }else {
                res.send(result);
              }
            });
    }
  });
});


//키워드 막대차트
app.post('/chart/bar', (req, res) => {
  let id = req.query.id;

  const sql = "SELECT keyword, count(keyword) as cnt FROM diary WHERE id=? GROUP BY keyword ORDER BY count(keyword) DESC LIMIT 5;"
  db.query(sql, id,
    (err, result) => {
        if (err)
            console.log(err);
        else
            res.send(result);
    });
})

//일기 통계
app.post('/chart/contribution',(req, res) => {
  let id = req.query.id;
  const sql= "SELECT CONCAT_WS('-', year, LPAD(month, 2, 0) , LPAD(day, 2, 0)) as date, 5 as count FROM diary WHERE id=?"

  db.query(sql, id,
    (err, result) => {
     console.log(result)
        if (err)
          res.send(err);
        else
        // console.log(result);
        res.send(result);
    });
});


//전체 감정 비율 반환 
app.post('/ratio', (req, res) => {
  let id = req.query.id;
  let values = [id];

  let avg  = [];

  const sql = "select COALESCE(sum(positive), 1) as positive, COALESCE(sum(negative), 1) as negative, COALESCE(sum(neutral), 1) as neutral, COALESCE(sum(positive) +  sum(negative) + sum(neutral), 3) as total From diary Where id = ?";
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }else {
      positive = result[0].positive;
      negative = result[0].negative;
      neutral = result[0].neutral;
      total = result[0].total;

      positive = positive/total*100;
      negative = negative/total*100;
      neutral = neutral/total*100;
      avg[0] = positive.toFixed();
      avg[1] = negative.toFixed();
      avg[2] = neutral.toFixed();

      
      res.send(avg);
    }
  });
});

//대표 감정 Pie차트
app.post('/pieTop', (req, res) => {
  let id = req.query.id;

  const sql = "SELECT top_emotion, COUNT(top_emotion) as count FROM diary WHERE id = ? GROUP BY top_emotion ORDER BY count DESC LIMIT 5";

  console.log("id: ",id)
  db.query(sql, id, (err, result) => {
    if(err){
      console.log("파이차트 에러: ", err)
    }else {
      console.log("파이차트 결과(대표 Top5): ", result);
      res.send(result);
    }
  });
});

//한해 감정 Line차트
app.post('/lineYear', (req, res) => {
  let id = req.query.id;
  let year = req.query.year;

  console.log("line_id: ", id);
  console.log("line_year: ", year);

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM (SELECT top_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND top_emotion IS NOT NULL UNION ALL SELECT second_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND second_emotion IS NOT NULL UNION ALL SELECT third_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND third_emotion IS NOT NULL) AS combined_table GROUP BY emotion_value ORDER BY count DESC LIMIT 5";
  let values = [id, year, id, year, id, year]
  db.query(sql, values, (err, result) => {
    if(err){
      console.log("라인차트 에러: ", err)
    }else {
      console.log("라인차트 결과(전체 Top5): ", result);
      res.send(result);
    }
  });
});


//한달 감정 Ring차트
app.post('/ringMonth', (req, res) => {
  let id = req.query.id;
  let month = req.query.month;

  console.log("ring_id: ", id);
  console.log("ring_month: ", month);

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM (SELECT top_emotion AS emotion_value FROM diary WHERE id = ? AND month = ? AND top_emotion IS NOT NULL UNION ALL SELECT second_emotion AS emotion_value FROM diary WHERE id = ? AND month = ? AND second_emotion IS NOT NULL UNION ALL SELECT third_emotion AS emotion_value FROM diary WHERE id = ? AND month = ? AND third_emotion IS NOT NULL) AS combined_table GROUP BY emotion_value ORDER BY count DESC LIMIT 4";
  let values = [id, month, id, month, id, month]
  db.query(sql, values, (err, result) => {
    if(err){
      console.log("링차트 에러: ", err)
    }else {
      console.log("링차트 결과(전체 Top5): ", result);
      res.send(result);
    }
  });
});


// // 사용자 일기내용 랜덤 5개
app.post('/randomDiary', (req, res) => {
  let id = req.query.id;

  let values = [id];
  
  let sql = "Select * From diary Where id = ? ORDER BY RAND() LIMIT 5;"
  db.query(sql, values, (err, result) => {
    if (err) {
      console.log(err);
    }
    else {
      res.send(result);
    }
  })
});

// 해당 일일다이어리 기준 week 데이터
app.post('/detailWeek', (req, res) => {
  let id = req.query.id;
  let year = req.query.year;
  let month = req.query.month;
  let day = req.query.day;

  const date = year + "-" + month + "-" + day
  console.log("date: ", date)

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM ( " +
    "SELECT top_emotion AS emotion_value " +
    "FROM diary " +
    "WHERE id = ? AND DATE(CONCAT(year, '-', month, '-', day)) BETWEEN DATE_SUB(?, INTERVAL WEEKDAY(?) DAY) AND DATE_ADD(?, INTERVAL 6 - WEEKDAY(?) DAY) " +
      "AND top_emotion IS NOT NULL " +
    "UNION ALL " +
    "SELECT second_emotion AS emotion_value " +
    "FROM diary " +
    "WHERE id = ? AND DATE(CONCAT(year, '-', month, '-', day)) BETWEEN DATE_SUB(?, INTERVAL WEEKDAY(?) DAY) AND DATE_ADD(?, INTERVAL 6 - WEEKDAY(?) DAY) " +
      "AND second_emotion IS NOT NULL " +
    "UNION ALL " +
    "SELECT third_emotion AS emotion " +
    "FROM diary " +
    "WHERE id = ? AND DATE(CONCAT(year, '-', month, '-', day)) BETWEEN DATE_SUB(?, INTERVAL WEEKDAY(?) DAY) AND DATE_ADD(?, INTERVAL 6 - WEEKDAY(?) DAY) " +
      "AND third_emotion IS NOT NULL " +
  ") AS combined_table " +
  "GROUP BY emotion_value";

  let values = [id, date, date, date, date, id, date, date, date, date, id, date, date, date, date]
  db.query(sql, values, (err, result) => {
    if (err) {
      console.log("week데이터 에러: ", err)
    } else {
      console.log("week데이터 결과(전체 Top5): ", result);
      res.send(result);
    }
  });
});


// 해당 일일다이어리 기준 month 데이터
app.post('/detailMonth', (req, res) => {
  let id = req.query.id;
  let year = req.query.year;
  let month = req.query.month;

  console.log("detailMonth_id: " + id)
  console.log("detailMonth_year: " + year)
  console.log("detailMonth_month: " + month)

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM (SELECT top_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND month = ? AND top_emotion IS NOT NULL UNION ALL SELECT second_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND month = ? AND second_emotion IS NOT NULL UNION ALL SELECT third_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? AND month = ? AND third_emotion IS NOT NULL) AS combined_table GROUP BY emotion_value ORDER BY count DESC";
  let values = [id, year, month, id, year, month, id, year, month]
  db.query(sql, values, (err, result) => {
    if (err) {
      console.log("month데이터 에러: ", err)
    } else {
      console.log("month데이터 결과(전체): ", result);
      res.send(result);
    }
  });
});
