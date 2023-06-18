## node_server 사용방법
본 프로젝트는 AWS ElasticBeanstalk 서비스를 이용하여, AWS 클라우드에 배포하였습니다.
<br><br><br>
## 로컬로 사용 시,
      - nodemon 설치 필요
      
      ```powershell
      npm install nodemon -g
      ```
      
      - 라이브러리 설치
      
      ```powershell
      npm install
      ```
      
      - 서버 실행
      
      ```powershell
      nodemon server.js
      ```
      
      - 프론트에서 ip 및 포트 번호 맞추기
      

<br><br><br>
## aws 사용 시,
  
  ```powershell
  npm install aws-sdk --save
  ```
  <br><br><br>
## aws 연동 (aws계정의 accessKeyId, secretAccessKey값 필요)
  
  ```jsx
  const s3 = new aws.S3({
    region: 'ap-northeast-2',
    accessKeyId: "---",
    secretAccessKey: "---"
  });
  ```
  <br><br><br>
## aws rds사용(sql 연동)
  
  ```jsx
  const db = mysql.createConnection({
    user: 'sodam',
    host: '---',
    password: '---',
    database: '---'
  });
  ```
  <br><br><br>
## aws s3연동
  
  ```jsx
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
      ```
