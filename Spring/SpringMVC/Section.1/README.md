# [1] 웹 서버, 웹 애플리케이션 이해

## 웹

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/204b3b73-e93b-4731-b7f2-1e8f9f4522bb/Untitled.png)

- HTTP 기반으로 통신. 모든것이 HTTP!
- 클라이언트에서 서버로 데이터를 전송할 때, 서버에서 클라이언트로 데이터를 응답할 때, 서버간 데이터 통신 등 모두 HTTP 프로토콜을 기반으로 동작

## 웹 서버 (Web Server)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/c98e2bf2-c641-4c84-abc7-96e8ea408063/Untitled.png)

- HTTP 기반으로 동작
- 정적 리소스 제공(html,css,js,이미지,같은 정적 데이터를 그냥 던져줌), 기타 부가기능
- 정적(파일) HTML, JSON, XML,CSS, JS, 이미지, 영상.. 전부 HTTP 기반으로 주고받음 (서버간의 데이터를 주고받을 때에도)
- 예) NGINX, APACHE

## 웹 애플리케이션 서버 (WAS)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/9e809c3a-1896-4f99-a768-d248049a0aec/Untitled.png)

- HTTP 기반으로 동작
- 웹 서버 기능 포함+ (정적 리소스 제공 가능)
- **프로그램 코드를 실행해서 애플리케이션 로직 수행 가능 (정적 리소스를 사용하는 웹 서버와의 차이점)**
    - 사용자의 요청에 따라 다른 화면 보여줄 수 있음 (프로그래밍이 가능)
    - 동적 HTML 생성, REST API=HTTP API(JSON) 제공
    - 서블릿, JSP, 스프링 MVC이 동작
- 예) 톰캣(Tomcat) Jetty, Undertow

## 웹 서버, 웹 애플리케이션 서버의 차이

- 웹 서버는 정적 리소스(파일), WAS는 애플리케이션 로직까지 실행 가능
- 사실은 둘의 용어도 경계도 모호함
    - 웹 서버도 프로그램을 실행하는 기능을 포함하기도 함
    - 웹 애플리케이션 서버도 웹 서버의 기능을 제공함
- 자바는 서블릿 컨테이너 기능을 제공하면 WAS라고 부름
    - 요즘은 서블릿 없이 자바코드를 실행하는 서버 프레임워크도 있음
- 정리하면, WAS는 애플리케이션 코드를 실행하는데 더 특화

- 실무에서 나눠지는 용도를 보자.

## 웹 시스템 구성 - WAS, DB

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/448ca2c9-b16b-4cf2-91a2-2af64d7d4aa3/Untitled.png)

- WAS, DB 만으로 최소한의 시스템 구성 가능
- WAS는 정적 리소스, 애플리케이션 로직 모두 제공 가능

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/ba17f1fd-c918-4a56-b20f-3df32115567b/Untitled.png)

- WAS 하나만 가지고 운영시 ⇒ WAS가 너무 많은 역할을 담당, 서버 과부하 우려
- 가장 비싼 애플리케이션 로직이 정적 리소스 때문에 수행이 어려울 수 있음
- (정적 리소스에 비해 애플리케이션 로직은 복잡한 과정을 거쳐 내려주어야 한다는 의미)
- WAS 장애시 접근조차 불가. 오류 화면조차도 노출 불가능
- 큰 시스템을 구축하기에는 부담이 있음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/c5514f1c-3c91-401d-a2fe-ef5a334e0675/Untitled.png)

- 가장 일반적인 방식
- 웹 서버를 앞에 두고 정적 리소스(html,css..)는 웹 서버가 처리
    - 업무 분담하여 부담 줄이기
- 웹 서버는 애플리케이션 로직같은 동적인 처리가 필요하면 WAS에 요청을 위임
- 장점) WAS는 중요한 애플리케이션 로직 처리 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/56f53339-8367-481e-ae82-7b172c248a57/Untitled.png)

- 효율적인 리소스 관리
    - 정적 리소스가 많이 사용되면 Web 서버 증설
    - 애플리케이션 리소스가 많이 사용되면 WAS 증설

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/9c872c61-d42b-4828-935b-390986b7ac5f/Untitled.png)

- 정적 리소스만 제공하는 웹 서버는 잘 죽지 않음 (복잡한 과정이 거의 없음)
- 애플리케이션 로직이 동작하는 WAS 서버는 잘 죽음
- WAS, DB 장애시에도 WEB 서버가 오류 화면 제공 가능

- CDN 같은 정적 리소스를 캐시할 수 있는 중간 서버를 놓기도 하고.. 이 구조를 기본 기반으로 발전
- 화면 제공이 아닌 API로 데이터만 제공하게 되면 웹 서버가 굳이 없어도 됨. WAS 서버만 구축해도 됨.

# [2] 서블릿

## HTML Form 데이터 전송

### POST 전송 - 저장

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/60153377-8c0e-48e9-ac0a-84729e1a37bf/Untitled.png)

- 회원가입 form
- html을 POST로 저장하기 위해 클라이언트가 POST로 서버에 전송 ⇒ 웹 브라우저가 요청 HTTP 메세지를 생성
- 데이터, 컨텐츠 타입(내용) 등의 정보가 있음

## 서버에서 처리해야 하는 업무

### 웹 애플리케이션 서버 직접 구현

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/0010b036-b903-4486-ba00-404e25970693/Untitled.png)

만약 웹 애플리케이션 서버(WAS)를 직접 구현해야 하면, 다음 프로세스를 거치도록 코드를 직접 작성해야 함

1. 먼저 서버에 TCP/IP 연결을 대기하고, 인터넷 망에서 연결이 들어오면 소켓을 연결하도록 함
2. HTTP 요청 메시지는 단순 텍스트이기 때문에, 파싱을 수행해 POST 방식과 /save URL을 인식
3. Content-Type을 확인하고, 이에 따라 메시지 body의 내용도 파싱해서 읽음
4. 저장에 대한 비즈니스 로직을 실행하고, DB에 저장 요청을 수행
    - 의미있는 비즈니스 로직
5. 저장 결과를 웹 브라우저에 전달하기 위해 HTTP 응답 메시지를 직접 생성
    - 시작 라인 생성, 헤더 생성, 메시지 body에 HTML 작성
6. TCP/IP로 HTTP 응답 메시지를 전달하고 소켓 종료

(4)만 의미있는 비즈니스 로직인데 전후로 단계가 너무 많음

- 전세계 개발자가 이 프로세스를 똑같이 개발하는 것은 너무 비효율적

→ 서블릿의 등장. (

### 서블릿을 지원하는 WAS 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/2e1b3199-7439-497b-a62d-cd3fe1f92d40/Untitled.png)

- 서블릿은 의미있는 비즈니스 로직(초록 상자)를 제외한 모든 일들을 지원해준다! 다 자동화

## 서블릿

### 특징

```java
@WebServlet(name = "helloServlet", urlPatterns = "/hello")
public class HelloServlet extends HttpServlet {
    @Override
    protected void service(HttpServletRequest request, HttpServletResponse response){
        //애플리케이션 로직
    }
}
```

- 웹 브라우저에서 요청이 왔을 때 urlPatterns(/hello)의 URL이 호출되면 서블릿 코드가 실행
- HttpServelet을 생속해서 사용
- HTTP 요청 정보를 편리하게 사용할 수 있는 HttpServletRequest를 사용 ⇒ HTTP 메세지의 값을 얻어낼 수 있음.
- HTTP 응답 정보를 편리하게 제공할 수 있는 HttpServletResponse ⇒ HTTP 응답 메세지를 생성하는 등의 편의 기능 제공.
- 개발자는 HTTP 스펙을 매우 편리하게 사용 ⇒ 기본적인 **HTTP 스펙을 어느 정도는 알아야 함..!**
    - Req, Res를 직접 파싱하려했다면 어려웠을 것

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/719b4588-776d-40bb-957a-43c242e85510/Untitled.png)

### HTTP 요청, 응답 흐름

- HTTP 요청시
    - WAS는 Request, Response 객체를 새로 만들어서 서블릿 객체 호출
    - 개발자는 Request 객체에서 HTTP 요청 정보를 편리하게 꺼내서 사용
    - 개발자는 Response 객체에 HTTP 응답 정보를 편리하게 입력
    - WAS는 Response 객체에 담겨있는 내용으로 HTTP 응답 정보를 생성→ 웹 브라우저에 전달

### 서블릿 컨테이너

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/8f67ec6f-3c33-4e00-8235-3ddc520f0c64/Untitled.png)

- 톰캣처럼 서블릿을 지원하는 WAS를 서블릿 컨테이너라고 함
- 개발자가 서블릿 컨테이너를 직접 생성하지 않아도 서블릿을 지원하는 WAS를 사용시
- 서블릿 컨테이너는 서블릿 객체를 자동으로 생성, 초기화, 호출, 종료하는 생명주기 관리
- 서블릿 객체는 **싱글톤**으로 관리
    - 객체를 하나만 생성해서 모두가 공유해서 사용하는 방식
    - 왜? Request-Response는 요청마다 항상 새롭게 생성해야함
    - 그러나, 고객의 요청이 올 때 마다 계속 서블릿 객체를 생성하는 것은 비효율
    - 최초 로딩 시점에 서블릿 객체를 미리 만들어두고 재활용
    - 모든 고객 요청은 동일한 서블릿 객체 인스턴스에 접근
    - **공유 변수 사용 주의**
    - 잘못 설정하면 다른 사용자가 내 정보를 보거나.. 할 수 있음
    - 서블릿 컨테이너 종료시 함께 종료
- JSP도 서블릿으로 변환 되어서 사용
- 동시 요청을 위한 **멀티 쓰레드 처리** 지원

# [3] 동시 요청 - 멀티 쓰레드

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/26cd31bb-9f5b-4a07-8e38-4fe777b48b69/Untitled.png)

## 쓰레드

- 애플리케이션 코드를 하나하나 순차적으로 실행하는 것은 쓰레드
- 자바 메인 메서드를 처음 실행하면 main이라는 이름의 쓰레드가 실행
- 쓰레드가 없다면 자바 애플리케이션 실행이 불가능
- 쓰레드는 한번에 하나의 코드 라인만 수행
- 동시 처리가 필요하면 쓰레드를 추가로 생성

## 단일 요청 - 쓰레드 하나 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/0b0dc095-ab1f-4ae1-bdd7-3a35ddbb4893/Untitled.png)

- WAS내에 쓰레드가 1개만 있다고 가정

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/39cd92d6-6eba-447b-8a96-027a30c7e9cb/Untitled.png)

- 쓰레드 연결 및 서블릿 호출

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/54c74b32-3387-4e42-8a73-f4ce02c404cf/Untitled.png)

- 서블릿 응답

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/bc20d19b-9ef2-4d62-8522-d2434ad757d7/Untitled.png)

- 응답을 수행한 뒤 쓰레드는 휴식

## 다중 요청 - 쓰레드 하나 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/50f27806-985a-47b2-8991-8c43ab9d43b4/Untitled.png)

- 쓰레드는 하나인데 요청이 여러개 들어오는 경우, 먼저 들어온 요청을 처리하기 위해 하나 있는 쓰레드가 할당되어 서블릿 코드를 실행
- 1번 요청 처리가 잘 수행되면, 순차적으로 2번 요청에 대한 응답이 수행하면 됨
- BUT, 1번 요청을 처리하는 과정에서 서블릿 코드 내부의 이유로 지연 발생

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/4654acb9-1714-4fe3-a4f8-fd90d740eaaf/Untitled.png)

- 지연 발생 중에 2번 요청이 들어오면, 쓰레드가 1개뿐이므로 대기

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/f35cfeb9-d295-4b7a-8430-f77f6927dbca/Untitled.png)

- 지연이 길어지면 1번에 대한 연결과 2번 요청에 대한 연결 모두 죽어버림
    - 2번 요청에 대한 수행 자체가 불가

## 요청마다 쓰레드 생성

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/274450bb-2f90-40ed-a3b5-955e5cb0d4e7/Untitled.png)

- 쓰레드를 하나 더 생성하고 동일한 서블릿을 호출해 2번 요청을 처리
    - 다른 쓰레드는 지연 중인 쓰레드와 별개로 코드를 실행한 뒤 응답을 수행

### 장단점

- 장점
    - 동시 요청을 처리할 수 있음
    - 리소스(CPU, 메모리)가 허용할 때 까지 처리가능
    - 하나의 쓰레드가 지연 되어도, 나머지 쓰레드는 정상 동작함
- 단점
    - 쓰레드는 생성 비용은 매우 비쌈
    - 고객의 요청이 올 때 마다 쓰레드를 생성하면, 응답 속도가 늦어짐
    - 쓰레드는 컨텍스트 스위칭 비용이 발생
    - 쓰레드 생성에 제한 X
    - 고객 요청이 너무 많이 오면, CPU, 메모리 임계점을 넘어서 서버가 죽을 수 있음

## 쓰레드 풀

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/c8934ad1-f06e-4e33-b729-eb9ed079d431/Untitled.png)

- 쓰레드가 필요할 때마다 쓰레드 풀에 요청하여, 이미 생성되어 있는 쓰레드를 받아 사용
- 쓰레드 사용을 종료하면 쓰레드 풀에 사용한 쓰레드를 반납
    - 쓰레드 종료가 아님

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/655517d1-7bf3-4484-bf0f-5df1252a94b1/Untitled.png)

- 최대 쓰레드가 모두 사용 중이라 쓰레드 풀에 쓰레드가 없다면, 요청을 거절하거나 특정 숫자만큼 대기하도록 설정할 수 있음

### 요청마다 쓰레드 생성의 단점 보완

- 특징
    - 필요한 쓰레드를 쓰레드 풀에 보관하고 관리함
    - 쓰레드 풀에 생성 가능한 쓰레드의 최대치를 관리함
        - 톰캣은 최대 200개 기본 설정 (변경 가능)
- 사용
    - 쓰레드가 필요하면, 이미 생성되어 있는 쓰레드를 쓰레드 풀에서 꺼내서 사용함
    - 사용을 종료하면 쓰레드 풀에 해당 쓰레드를 반납함
    - 최대 쓰레드가 모두 사용중이어서 쓰레드 풀에 쓰레드가 없으면?
        - 기다리는 요청은 거절하거나 특정 숫자만큼만 대기하도록 설정할 수 있음
- 장점
    - 쓰레드가 미리 생성되어 있으므로, 쓰레드를 생성하고 종료하는 비용(CPU)이 절약되고, 응답 시간이 빠르다
    - 생성 가능한 쓰레드의 최대치가 있으므로 너무 많은 요청이 들어와도 기존 요청은 안전하게 처리할 수 있다
    

### 실무 팁

- WAS의 주요 튜닝 포인트는 최대 쓰레드(max thread) 수이다
- 이 값을 너무 낮게 설정하면?
    - 동시 요청이 많으면, 서버 리소스는 여유롭지만, 클라이언트는 금방 응답 지연
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/fee20204-e4d4-401e-815a-cee355c6f2f3/Untitled.png)
    
    - 최대 쓰레드 수를 10으로 설정한다면, 동시에 100개의 요청이 오는 경우 10개 요청만 처리하고 90개 요청은 대기 or 거절
    - 10개의 요청 처리 시, CPU의 5%만 사용한다면 매우 비효율
        - WAS는 살아있을지 몰라도 서비스에는 장애 발생
            - 못해도 CPU 사용율은 50%를 넘겨야 개발자가 어느정도 제대로 세팅했다고 볼 수 있음
- 이 값을 너무 높게 설정하면?
    - 동시 요청이 많으면, CPU, 메모리 리소스 임계점 초과로 서버 다운
- 장애 발생시?
    - 클라우드면 일단 서버부터 늘리고, 이후에 튜닝
    - 클라우드가 아니면 열심히 튜닝

### 쓰레드 풀의 적정 숫자

- 애플리케이션 로직의 복잡도, CPU, 메모리, IO 리소스 상황에 따라 모두 다름
- 성능 테스트
    - 최대한 실제 서비스와 유사하게 성능 테스트 시도
    - 툴: 아파치 ab, 제이미터, nGrinder

## WAS의 멀티 쓰레드 지원

- 멀티 쓰레드에 대한 부분은 WAS가 처리
- 개발자가 멀티 쓰레드 관련 코드를 신경쓰지 않아도 됨
- 개발자는 마치 싱글 쓰레드 프로그래밍을 하듯이 편리하게 소스 코드를 개발
- 멀티 쓰레드 환경이므로 싱글톤 객체(서블릿, 스프링 빈)는 주의해서 사용

# [4] HTML, HTTP API, CSR, SSR

## 정적 리소스

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/0a92c1e6-af0b-4418-ad28-3255c44eab4e/Untitled.png)

- 고정된 HTML 파일, CSS, JS, 이미지, 영상 등을 제공
- 주로 웹 브라우저

## HTML 페이지

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/49430e48-60be-4114-9f09-d8bc6137688c/Untitled.png)

- 동적으로 필요한 HTML 파일을 생성해서 전달
- 웹 브라우저: HTML 해석

## HTTP API

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/35f391f2-a21d-46da-8948-86e021755652/Untitled.png)

- HTML이 아니라 데이터를 전달
- 주로 JSON 형식 사용
- 다양한 시스템에서 호출

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/fbad3bd2-e91d-403d-9804-64a595c24eb3/Untitled.png)

- 주로 JSON 형태로 데이터 통신
- 앱 클라이언트
    - 데이터를 받아 앱의 UI  컴포넌트를 이용해 사용자에게 뷰 제공
- 웹 클라이언트
    - 데이터를 받아 React, Vue.js같은 웹 클라이언트를 이용해 사용자에게 뷰 제공
- 서버 to 서버
    - 주문 서버와 결제 서버가 통신할 때, HTML 주고받을 필요 없이 데이터만 주고 받음
    - 요즘은 마이크로서비스가 유행 → 기업 안에서도 여러 서비스들이 쪼개져 있고, 각 서버는 HTTP API로 통신
    - 기업 간 데이터 통신에도 사용함

⇒ 백엔드 개발자는 정적 리소스, HTML 페이지, HTTP API 이 3가지를 어떻게 제공할 지 고민해야 한다

## 서버사이드 렌더링, 클라이언트 사이드 렌더링

- SSR - 서버 사이드 렌더링
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/db855c66-1d13-47be-b093-2c87456b4792/Untitled.png)
    
    - HTML 최종 결과를 서버에서 만들어서 웹 브라우저에 전달
    - 주로 정적인 화면에 사용
    - 관련기술: JSP, 타임리프 → 백엔드 개발자

- CSR - 클라이언트 사이드 렌더링
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b4218a-e3b3-4354-935e-87ebfb47259e/795dda17-ad33-4026-8475-ad11680add21/Untitled.png)
    
    - HTML 결과를 자바스크립트를 사용해 웹 브라우저에서 동적으로 생성해서 적용
    - 주로 동적인 화면에 사용, 웹 환경을 마치 앱 처럼 필요한 부분부분 변경할 수 있음
    - 예) 구글 지도(확대, 축소해도 url이 변하지 않고 앱처럼 동작), Gmail, 구글 캘린더
    - 관련기술: React, Vue.js → 웹 프론트엔드 개발자
    
- 참고
    - React, Vue.js를 CSR + SSR 동시에 지원하는 웹 프레임워크도 있음
    - SSR을 사용하더라도, 자바스크립트를 사용해서 화면 일부를 동적으로 변경 가능

## 백엔드 개발자 입장에서 UI 기술

- 백엔드 - 서버 사이드 렌더링 기술
    - JSP, 타임리프
    - 화면이 정적이고, 복잡하지 않을 때 사용
    - 백엔드 개발자는 서버 사이드 렌더링 기술 학습 필수
- 웹 프론트엔드 - 클라이언트 사이드 렌더링 기술
    - React, Vue.js
    - 복잡하고 동적인 UI 사용
    - 웹 프론트엔드 개발자의 전문 분야
- 선택과 집중
    - 백엔드 개발자의 웹 프론트엔드 기술 학습은 옵션
    - 백엔드 개발자는 서버, DB, 인프라 등등 수 많은 백엔드 기술을 공부해야 한다
    - 웹 프론트엔드도 깊이있게 잘 하려면 숙련에 오랜 시간이 필요하다