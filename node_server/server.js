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
app.use(cors());


// aws 연동
const s3 = new aws.S3({
  region: '---',
  accessKeyId: "---",
  secretAccessKey: "---"
});

// sql 연동
const db = mysql.createConnection({
  user: '---',
  host: '---',
  password: '---',
  database: '---'
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


// //플라스크 로컬 통신
// app.post('/flask', (req, res) => {
//   axios({
//     method: "POST",
//     url: "http://192.168.0.18:5000/getSentiment",
//     data: {
//       input:"나 너무 살쪘어ㅜㅜ"
//     },
//   })
//   .then((result) => {
//     console.log(result);
//     res.send(result);
//   })
// })



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
          service_id: "---",
          template_id: "---",
          user_id: "---",
          accessToken: "---",
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
  console.log("내 이름은 이재문 천재죠.");
  let text = req.query.text;
  axios({
    method: "POST",
    url: "http://192.168.0.18:5000/getSentiment",
    data: {
      input: text
    },
  })
  .then((result) => {
    console.log(result.data.sentence);
    console.log(result.data.emotion);
    console.log(result.data.index);
    res.send(result.data);
  })
  .catch((error) => {
    console.error(error);
    res.status(500).send(error);
  });
});


//글 작성
app.post('/write', async function(req, res) {
  console.log("글 작성 하러 옴");
  let id = req.query.id;
  let title = req.query.title;
  let content = req.query.content;
  let year = req.query.year;
  let month = req.query.month;
  let day = req.query.day;
  let img = req.query.img;
  let cb_sentence = req.query.cb_sentence;
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

  setTimeout(() => {
    let values = [id, title, content, year, month, day, img, cb_sentence, cb_emotion, top_emotion, second_emotion, third_emotion, top_number, second_number, third_number];
    const sql = "INSERT INTO diary(id, title, content, year, month, day, img, cb_sentence, cb_emotion, top_emotion, second_emotion, third_emotion, top_number, second_number, third_number) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"


    db.query(sql, values, (err, result) => {
      if (err) {
        console.log(err);
      } else {
        res.send(result)
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




  let values = [title, content, year, month, day, img, diarykey];
  console.log(values);
  const sql = "Update diary Set title = ?, content = ?, year = ?, month = ? , day = ?, img = ? Where diarykey = ?"

  db.query(sql, values,
    (err, result) => {
        if (err)
            console.log(err);
        else
            res.send(result);
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

  let values = [diarykey];
  let sql = "Select title, content, year, month, day, img From diary Where diarykey =?"
  db.query(sql, values, (err, result) => {
    if(err) {
      console.log(err);
    }
  else {
      res.send(result);
    }
})
});


//앨범
app.post('/album', (req, res) => {
  console.log("앨범");
  let id = req.query.id;
  let values = [id];

  let sql = "select diarykey, img From diary Where id =?"
  db.query(sql, values, (err, result) => {
    if(err) {
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
app.post('/PieTop', (req, res) => {
  let id = req.query.id;

  const sql = "SELECT top_emotion, COUNT(top_emotion) as count FROM diary WHERE id = '?' GROUP BY top_emotion ORDER BY count DESC LIMIT 5";

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
app.post('/LineYear', (req, res) => {
  let id = req.query.id;
  let year = req.query.year;

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM (SELECT top_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? UNION ALL SELECT second_emotion AS emotion_value FROM diary WHERE id = ? AND year = ? UNION ALL SELECT third_emotion AS emotion_value FROM diary WHERE id = ? AND year = ?) AS combined_table GROUP BY emotion_value ORDER BY count DESC LIMIT 5;";
  values[id, year, id, year, id, year]
  db.query(sql, values, (err, result) => {
    if(err){
      console.log("파이차트 에러: ", err)
    }else {
      console.log("라인차트 결과(전체 Top5): ", result);
      res.send(result);
    }
  });
});


//한달 감정 Ring차트
app.post('/RingMonth', (req, res) => {
  let id = req.query.id;
  let month = req.query.month;

  const sql = "SELECT emotion_value, COUNT(*) AS count FROM (SELECT top_emotion AS emotion_value FROM diary WHERE id = ? AND month = ? UNION ALL SELECT second_emotion AS emotion_value FROM diary WHERE id = ? AND month = ? UNION ALL SELECT third_emotion AS emotion_value FROM diary WHERE id = ? AND month = ?) AS combined_table GROUP BY emotion_value ORDER BY count DESC LIMIT 5;";
  values[id, month, id, month, id, month]
  db.query(sql, values, (err, result) => {
    if(err){
      console.log("파이차트 에러: ", err)
    }else {
      console.log("라인차트 결과(전체 Top5): ", result);
      res.send(result);
    }
  });
});

