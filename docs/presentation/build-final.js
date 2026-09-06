// Черновик защиты: титул + 4 слайда. Минимальный кегль — 30.
// Ёмкость строки при 30 pt Arial: ~26 знаков на карточке 5,5" и ~59 во всю ширину.
// Высота строки 0,5". Числа сверены с репозиторием на 06.09.2026.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
const W = 13.333, H = 7.5;

const PURPLE="6C2BF5", RED="EF3124", GRAY="EBEBEB", WHITE="FFFFFF";
const PINK="FDEAE8", LIME="C3F53C", INK="111111", MUTED="5E5E5E";
const F="Arial", LH=0.50;

function badge(s,x,y,sc){ sc=sc||1;
  s.addShape(p.ShapeType.rect,{x,y,w:1.85*sc,h:0.62*sc,fill:{color:RED},rotate:-5});
  s.addText("АЛЬФА",{x,y:y+0.02*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
  s.addText("БУДУЩЕЕ",{x,y:y+0.29*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
}
function slide(red,black){
  const s=p.addSlide(); s.background={color:GRAY};
  badge(s,W-2.05,0.24,0.8);
  s.addText([{text:red,options:{color:RED}},{text:black?" "+black:"",options:{color:INK}}],
    {x:0.5,y:0.28,w:10.5,h:0.7,fontFace:F,fontSize:34,bold:true,isTextBox:true,margin:0});
  return s;
}
function card(s,x,y,w,h,fill){ s.addShape(p.ShapeType.roundRect,{x,y,w,h,fill:{color:fill||WHITE},rectRadius:0.14,
  shadow:{type:"outer",color:"9A9A9A",blur:8,offset:1,angle:90,opacity:0.18}}); }
function txt(s,t,x,y,w,n,size,col,bold){
  s.addText(t,{x,y,w,h:n*LH,fontFace:F,fontSize:size||30,color:col||"3A3A3A",bold:!!bold,
    isTextBox:true,margin:0,lineSpacingMultiple:1.0,valign:"top"});
}

/* ---------- 0. Титул ---------- */
{
  const s=p.addSlide(); s.background={color:PURPLE};
  badge(s,0.55,0.45,1.2);
  s.addShape(p.ShapeType.roundRect,{x:W-2.6,y:0.5,w:2.0,h:0.55,fill:{color:RED},rectRadius:0.27});
  s.addText("Альфа Банк",{x:W-2.6,y:0.5,w:2.0,h:0.55,fontFace:F,fontSize:16,bold:true,color:WHITE,align:"center",valign:"middle",isTextBox:true,margin:0});
  s.addText("Триггерная модель\nдля трансграничных переводов",
    {x:0.7,y:2.6,w:11.9,h:2.0,fontFace:F,fontSize:48,bold:true,color:WHITE,isTextBox:true,margin:0,lineSpacingMultiple:1.05});
  s.addText("Пуш о моменте для перевода, под капотом ML-модель",
    {x:0.72,y:4.8,w:11.9,h:0.55,fontFace:F,fontSize:30,color:"E4D8FF",isTextBox:true,margin:0});
  s.addText("Сырых Ольга · Бикинеев Амир · Немкин Кирилл",
    {x:0.72,y:5.95,w:11.9,h:0.5,fontFace:F,fontSize:22,color:"C9B4FF",isTextBox:true,margin:0});
}

/* ---------- 1. Проблема ---------- */
{
  const s=slide("Проблема","и кто с ней живёт");
  card(s,0.5,1.2,6.05,2.5,WHITE);
  txt(s,"Трудовой мигрант",0.8,1.34,5.45,1,30,RED,true);
  txt(s,"Таджикистан, Узбекистан\nи Киргизия — 87 % всей\nтрудовой миграции в РФ",0.8,1.9,5.45,3,30,"3A3A3A");

  card(s,6.78,1.2,6.05,2.5,WHITE);
  txt(s,"Как он переводит",7.08,1.34,5.45,1,30,RED,true);
  txt(s,"Семье, раз в месяц после\nрасчёта. В приложении\nне видно, сколько дойдёт",7.08,1.9,5.45,3,30,"3A3A3A");

  card(s,0.5,3.85,12.33,1.5,PINK);
  s.addText("2 200 ₽",{x:0.85,y:3.99,w:3.4,h:0.8,fontFace:F,fontSize:44,bold:true,color:RED,isTextBox:true,margin:0});
  txt(s,"между лучшим и худшим днём месяца",4.5,4.03,8.2,1,30,INK,true);
  txt(s,"на среднем переводе в 45 000 ₽",4.5,4.53,8.2,1,30,"3A3A3A");

  card(s,0.5,5.5,12.33,1.45,WHITE);
  txt(s,"Альтернативы: следить за курсом самому · не делать ничего ·",0.85,5.63,11.6,1,26,"3A3A3A");
  txt(s,"алерты Wise и Xe, где цель ставит сам клиент",0.85,6.05,11.6,1,26,"3A3A3A");
  txt(s,"Порог задаёт клиент. Момент за него не выбирает никто",0.85,6.47,11.6,1,26,INK,true);
}

/* ---------- 2. Сравнение коридоров ---------- */
{
  const s=slide("Какой коридор","и почему");
  const head=["Коридор","Uplift","В неделю","Клиенту","Рынок"];
  const rows=[["Узбекистан","1,56","0,61","1 053 ₽","608 млн"],
              ["Таджикистан","1,44","1,14","416 ₽","248 млн"],
              ["Киргизия","1,38","1,11","341 ₽","244 млн"],
              ["Армения","1,44","1,11","450 ₽","94 млн"],
              ["Казахстан","1,30","1,11","153 ₽","62 млн"]];
  const tbl=[head.map(x=>({text:x,options:{bold:true,color:WHITE,fill:{color:INK}}}))]
    .concat(rows.map((r,i)=>r.map(x=>({text:x,options:{bold:i===0,color:i===0?INK:"3A3A3A",
      fill:{color:i===0?LIME:WHITE}}}))));
  s.addTable(tbl,{x:0.5,y:1.2,w:12.33,colW:[3.0,1.7,2.3,2.4,2.93],fontFace:F,fontSize:30,
    border:{type:"solid",color:"D6D6D6",pt:1},rowH:0.5,valign:"middle"});

  card(s,0.5,5.05,12.33,1.9,WHITE);
  txt(s,"Узбекистан: лучший uplift, вдвое клиенту, вшестеро рынок",0.85,5.16,11.6,1,26,INK,true);
  txt(s,"Клиенту — за год при 9 переводах по 45 000 ₽",0.85,5.66,11.6,1,24,"3A3A3A");
  txt(s,"Рынок — весь коридор: операции × чек × 5 % прироста × 2 % маржи",0.85,6.08,11.6,1,24,"3A3A3A");
  txt(s,"Армения и Казахстан — другой отправитель, не трудовая миграция",0.85,6.50,11.6,1,24,"3A3A3A");
}

/* ---------- 3. Как устроено ---------- */
{
  const s=slide("Как это","устроено");
  const col=[["Данные","Курсы ЦБ за 8 лет\nБиржа и нацбанки\nВсё открытое\nи воспроизводимое"],
             ["Модель","CatBoost, 5 дней\nПодтверждение\nпятью коридорами\n175 вариантов"],
             ["Метрики","Uplift 1,56\nТочность 62 %\nВыгода 26 бп\nБез заглядывания"]];
  col.forEach((c,i)=>{ const x=0.5+i*4.28;
    card(s,x,1.2,4.05,2.8,WHITE);
    txt(s,c[0],x+0.28,1.34,3.5,1,30,RED,true);
    txt(s,c[1],x+0.28,1.9,3.6,4,26,"3A3A3A");
  });
  card(s,0.5,4.15,8.05,2.8,PINK);
  txt(s,"Что решение не умеет",0.8,4.28,7.45,1,30,INK,true);
  txt(s,"Не ловит дно: берём 26 бп из 129\n\nСтрогая мера uplift 1,25, не 1,56\n\nКурс ЦБ — не курс сделки, спред 9 %",0.8,4.9,7.45,5,24,"3A3A3A");

  card(s,8.78,4.15,4.05,2.8,WHITE);
  txt(s,"Команда",9.06,4.28,3.5,1,30,RED,true);
  txt(s,"Ольга — данные и ML\nАмир — ML\nКирилл — продукт",9.06,4.82,3.55,3,22,"3A3A3A");
  txt(s,"Персоны в прогоне:\nАзиз, Далер, Армен",9.06,6.05,3.55,2,22,INK,true);
}

/* ---------- 4. Готовность и пилот ---------- */
{
  const s=slide("Готовность","и следующий шаг");
  card(s,0.5,1.2,6.05,2.9,WHITE);
  txt(s,"Готово",0.8,1.34,5.45,1,30,RED,true);
  txt(s,"Сигналы и бэктест\nРасчёт на любую дату\nПрототип пути\nТексты на персонах",0.8,1.9,5.45,4,30,"3A3A3A");

  card(s,6.78,1.2,6.05,2.9,WHITE);
  txt(s,"Пилот — A/B на клиентах",7.08,1.34,5.45,1,30,RED,true);
  txt(s,"20 000 клиентов в группу\nСигнал виден за 3 недели\nПоведение — за квартал\nПервым проверяем спред",7.08,1.9,5.45,4,30,"3A3A3A");

  card(s,0.5,4.3,12.33,1.75,PINK);
  txt(s,"Успех — переводов стало значимо больше",0.85,4.44,11.6,1,30,INK,true);
  txt(s,"Порог различимости при 20 000 в группе — 2,9 %.",0.85,5.0,11.6,1,26,"3A3A3A");
  txt(s,"Меньше не отличим от шума и успехом не считаем",0.85,5.46,11.6,1,26,"3A3A3A");

  s.addShape(p.ShapeType.roundRect,{x:0.5,y:6.2,w:12.33,h:0.75,fill:{color:LIME},rectRadius:0.14});
  s.addText("Не предсказываем курс. Показываем сумму",
    {x:0.85,y:6.2,w:11.6,h:0.75,fontFace:F,fontSize:30,bold:true,color:INK,valign:"middle",isTextBox:true,margin:0});
}

p.writeFile({ fileName: "fx-pulse-final-draft.pptx" }).then(f => console.log("Готово:", f));
