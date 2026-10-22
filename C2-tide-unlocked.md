# 2026 — Challenge 2 (ĐỀ THẬT): Tide "UNLOCKED"

Trang đề: https://tide.org/unlocked/ · đọc trực tiếp 22/10/2026
Nhà tài trợ: **Tide** (Industry Challenge Sponsor). Đây là **Industry Challenge** — tách riêng khỏi Challenge 1 trên Kaggle.
⏰ **Hạn nộp: Thứ Năm 22/10, 18:00** (giờ Việt Nam và Malaysia) — cùng giờ với Challenge 1. Lúc đọc còn ~8 tiếng 40 phút.
Khẩu hiệu: *"What would you build if trust stopped being the bottleneck?"* — **Không cần kinh nghiệm an ninh mạng hay mật mã.**

## 1. Đề là gì?
Xây **một prototype nhỏ chạy được, có tích hợp TideCloak**, thể hiện **một khả năng mới** mà hôm nay bị cản trở vì vấn đề *niềm tin / quyền riêng tư / bảo mật*.

Ý tưởng cốt lõi: nhiều ý tưởng lớn đòi hỏi trao **quá nhiều quyền** cho một ai đó. Bệnh viện muốn chia sẻ hồ sơ chung — nhưng ai kiểm soát? AI làm được nhiều hơn — nhưng cho nó bao nhiêu quyền truy cập? Cộng đồng muốn quản trị tài nguyên chung — nhưng ai quyết định cuối cùng?

→ Chọn **một khả năng đang bị kẹt vì trade-off đó**. Dùng TideCloak để xây app mà **không một admin / tổ chức / AI nào có thể đơn phương ghi đè** các luật bảo vệ dữ liệu và quyết định của người dùng. Cho thấy điều đó mở ra khả năng gì mới.

> *"Bring a working prototype with TideCloak integrated. Demonstrate one new possibility and explain why removing unilateral control makes it possible."*

## 2. Công cụ: TideCloak
- TideCloak là **hệ thống drop-in** cho: xác thực người dùng, phân quyền, bảo vệ dữ liệu.
- Nó dùng **Emergent Authority**: cho phép app thực hiện hành động nhạy cảm **mà không trao cho bất kỳ admin / server / vendor nào quyền đơn phương ghi đè** các lớp bảo vệ.
- **Emergent Authority** = không ai một mình giữ "chìa khóa chủ". Quyền chỉ "nổi lên" khi đúng *identity + policy + approvals* khớp nhau.

⚠️ Cảnh báo chấm điểm: *"If your app would work exactly the same with ordinary login and backend permissions, develop the idea further. TideCloak should change what it can credibly do."* — Nếu app chạy y hệt với login + phân quyền backend thông thường thì chưa đạt. TideCloak phải **thay đổi được điều app có thể làm một cách đáng tin cậy**.

## 3. Bốn bước từ ý tưởng đến demo
1. **Chọn một khả năng** — một thứ hữu ích đang bị kẹt vì trust/privacy/security.
2. **Kết nối AI coding agent** — thêm **Tide's MCP server** vào agent (Claude Code, Cursor, VS Code, hoặc agent MCP-compatible khác). MCP cấp cho agent công cụ + chỉ dẫn để tích hợp TideCloak.
   - Plan B nếu MCP trục trặc: dùng **TideCloak Quickstart** (template site đã tích hợp sẵn TideCloak, build ý tưởng lên trên; chạy với bất kỳ model/editor nào, hoặc không cần AI).
3. **Build với TideCloak** — mô tả app và luật của nó cho agent. Dùng **synthetic data và hệ thống mock**.
4. **Demo và nộp** — quay video 2 phút cho thấy khả năng mới chạy được + vai trò của TideCloak.

> *"You don't need a finished product. Focus on one convincing demonstration."* · Dùng data giả, mock hệ thống bên ngoài — không cần hoàn chỉnh.

## 4. Nộp gì? (2 thứ, đều trong 1 repo GitHub public, nộp link qua Google Form)

### (1) Repo GitHub public
Code prototype có tích hợp TideCloak + video demo. **Để repo public.**

**README bắt buộc** — copy template của Tide lên đầu README, thêm 1 câu mô tả, rồi trả lời 5 câu (khớp đúng tiêu chí chấm). **≤ 300 từ.** Hướng dẫn cài đặt để ở section riêng.

5 câu trong README:
1. **Vision & originality** — Bạn đề xuất sản phẩm/dịch vụ/trải nghiệm/cách làm việc mới nào? Điều gì khiến nó khác biệt?
2. **Trust constraint insight** — Hôm nay điều gì khó/rủi ro/bất khả thi vì trust/privacy/security cản đường? Gỡ bỏ ràng buộc đó mở ra điều gì?
3. **Value & impact** — Ai hưởng lợi, cuộc sống/công việc họ cải thiện ra sao? Cho ví dụ cụ thể.
4. **Why TideCloak matters** — App dùng TideCloak thế nào, và tích hợp đó giúp hiện thực hóa ý tưởng ra sao?
5. **Prototype & storytelling** — Prototype hôm nay thể hiện được gì? Tương tác chính giám khảo cần chú ý là gì?

### (2) Video demo 2 phút
Cho thấy khả năng chạy được. Lưu file **.mp4 trong repo GitHub, < 100 MB** (giới hạn file của GitHub).
Cấu trúc gợi ý: **The possibility** (chọn build gì) → **The constraint** (vấn đề trust nào cản) → **The prototype** (chạy được + vai trò TideCloak) → **What changes** (giờ người dùng làm được gì mới).

## 5. Tiêu chí chấm (tổng 100%)
| Tiêu chí | % | Nội dung |
|---|---|---|
| **Vision & originality** | 25 | Câu trả lời tươi mới, tham vọng cho thử thách |
| **Trust constraint insight** | 20 | Chỉ rõ chính xác trust/privacy/custody/authority nào đang cản hôm nay |
| **Value & impact** | 20 | Tạo giá trị thật cho người/tổ chức/xã hội |
| **Why TideCloak matters** | 20 | TideCloak thay đổi thực chất điều sản phẩm có thể làm đáng tin |
| **Prototype & storytelling** | 15 | Làm ý tưởng dễ hiểu và cho thấy nó chạy |

Giám khảo đọc README project overview **song song với** video demo.

## 6. 12 ý tưởng gợi ý (không bắt buộc, build ở bất kỳ lĩnh vực nào)
Một vài ví dụ Tide đưa ra:
- **Care Passport** (Y tế) — chia sẻ hồ sơ sức khỏe chọn lọc với bác sĩ cụ thể, bệnh nhân kiểm soát quyền truy cập.
- **Research Consent Vault** (Nghiên cứu) — người tham gia chia sẻ dữ liệu với nhà nghiên cứu được duyệt, rút quyền truy cập tương lai.
- **Rental Application Passport** (Nhà ở) — tái dùng hồ sơ thuê nhà, chỉ tiết lộ đúng thứ mỗi chủ nhà cần.
- **Blind Portfolio Review** (Giáo dục) — chấm bài trước khi lộ danh tính, quyền truy cập theo giai đoạn review.
(Trang có nút "Show all 12 ideas".)

## 7. Tài nguyên
- **Watch the launch** / **Watch the walkthrough** (Tide's MCP with Claude Code) — video trên trang.
- **Ask Hackbot** — trợ lý AI của Tide, hỏi thẳng về ý tưởng/README/code.
- Blog: *"Beyond cybersecurity: how to think about Emergent Authority"* (start here) và *"How Tide works: the RMIT presentation"*.
- **Connect your coding agent** (MCP) · **TideCloak Quickstart** (Plan B) · **TideCloak overview/documentation**.

## 8. Khác biệt lớn so với hình dung trước
Challenge 2 **KHÔNG phải task ML/Kaggle**. Đây là task **xây app full-stack + tích hợp TideCloak + kể chuyện**, hợp với người mạnh web/product hơn người mạnh ML. Hai challenge chạy song song cùng hạn 18:00 → **chia người**: nhóm ML lo Challenge 1 (Kaggle), nhóm web/vibe-coding lo Challenge 2 (Tide).

## 9. Chiến thuật trong ~8 tiếng
1. **Chọn ý tưởng trong 20 phút đầu** — lấy 1 trong 12 gợi ý hoặc biến thể, miễn là *chỉ khả thi khi bỏ quyền đơn phương*.
2. Cài **Tide MCP** vào Claude Code / Cursor ngay (hoặc dùng TideCloak Quickstart nếu MCP lỗi).
3. Build prototype tối giản với **data giả + mock** — chỉ cần 1 luồng demo thuyết phục, không cần hoàn chỉnh.
4. Viết README ≤ 300 từ theo đúng 5 prompt = đúng 5 tiêu chí chấm.
5. **Quay video 2 phút sớm** (dành ≥ 1 tiếng cuối): possibility → constraint → prototype (vai trò TideCloak) → what changes. Lưu .mp4 < 100MB trong repo.
6. Để repo **public**, nộp link qua **Google Form** trước 18:00.
7. Thử "thay TideCloak bằng login thường" — nếu app vẫn chạy y hệt thì **đào sâu ý tưởng hơn**, nếu không sẽ mất phần lớn 20% "Why TideCloak matters".
